from together import Together
import qdrant_client

import os
import tqdm
import json
import numpy as np

from llama_index.embeddings.fastembed import FastEmbedEmbedding

from io import StringIO

import traceback as tb

import argparse

parser = argparse.ArgumentParser(description="Process file paths for MultiHopRAG.")

parser.add_argument("-r","--root", type=str, required=True, help="Root directory path.")
parser.add_argument("-o", "--output_file_name", type=str, required=True, help="Output file name.")
parser.add_argument("-c", "--chunk_size", type=int, default=500, help="Chunk size for splitting text.")
parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", help="model name.")

parser.add_argument("--database_file", type=str, default="multihoprag/corpus.json", help="Database file relative to root.")
parser.add_argument("--query_file", type=str, default="multihoprag/MultiHopRAG.json", help="Query file relative to root.")
parser.add_argument("--query_start_id", type=int, default=0, help="Query start id.")


args = parser.parse_args()

#--------------------------------------------------------------------------
""" Argument preprocessing """
root = args.root #"/scratch1/rachital/nlp"

database_file = os.path.join(root, args.database_file)
query_file = os.path.join(root, args.query_file)

output_tsv_file = os.path.join(root, args.output_file_name + ".tsv")
output_json_file = os.path.join(root, args.output_file_name + ".json")

#--------------------------------------------------------------------------

""" Connect to Qdrant Cloud """
quadrant_client = qdrant_client.QdrantClient(
    url=os.getenv("QUADRANT_DB_URL"),  # Quadrant DB URL
    api_key=os.getenv("QUADRANT_API_KEY")
)

embed_model = FastEmbedEmbedding(model_name="BAAI/bge-large-en-v1.5")

#--------------------------------------------------------------------------

""" Load and process dataset """

# Script to index documents as per titles
with open(database_file, 'r') as f:
  database = json.load(f)

doc2id = {article["title"]:i for i,article in enumerate(database)}

# Script to get gold retrieval indices for every query

with open(query_file, 'r') as f:
  query_samples = json.load(f)

for query in query_samples:
  gold_retrievals = []
  for evidence in query["evidence_list"]:
    gold_retrievals.append(doc2id[evidence["title"]])
  query["gold_evidence"] = gold_retrievals

#--------------------------------------------------------------------------

def search_similar(query):
    query_embedding = embed_model.get_text_embedding(query)
    
    # Search in Qdrant
    search_result = quadrant_client.search(
        collection_name="bge-large-256-embedds",
        query_vector=query_embedding,
        limit=10
    )
    data_list = []
    
    for i,point in enumerate(search_result):
        data_list.append({
            "text": f"[Evidence {i+1}] "
                    # f"[Excerpt from document]\n"
                    f"Title: {point.payload.get('title')}\n"
                    # f"published_at: {point.payload.get('published_at')}\n"
                    # f"source: {point.payload.get('source')}\n"
                    f"Body: {json.load(StringIO(point.payload.get('_node_content')))['text']}",
            "score": point.score
        })
    return data_list


class ChainOfAgents():
    def __init__(self, client, model, chunk_size = 1500, temperature=0.7):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.chunk_size = chunk_size
    
    def gpt3_completion(self, prompt, role):
        if role == "worker":
            system_prompt = "You are an AI assistant that performs multi-hop reasoning based on multiple pieces of evidence. Summarize the context you have that might hold the answer."
        elif role == "manager":
            system_prompt = "You are an AI assistant that performs multi-hop reasoning based on multiple pieces of evidence. Provide one word answer."

        response = self.client.chat.completions.create(
        model=self.model,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content

    def worker_agent(self, chunk, task_prompt):
        """
        Process a chunk of text with a specific task prompt.
        """

        prompt = f"Question: {task_prompt}\n\n" + \
                f"Context: {chunk}" + \
                f"Answer:"
        # prompt = f"{task_prompt}\n\n{chunk}"
        return prompt, self.gpt3_completion(prompt, role="worker")

    def manager_agent(self, worker_outputs, task_prompt, synthesis_prompt):
        """
        Synthesize the outputs from worker agents into a final response.
        """
        combined_output = "\n\n".join(worker_outputs)

        prompt = f"{synthesis_prompt}\n\n" + \
                f"Question: {task_prompt}\n\n" + \
                f"Context: {combined_output}\n" + \
                f"Answer:"
        # prompt = f"{synthesis_prompt}\n\n{combined_output}"
        return prompt, self.gpt3_completion(prompt, role="manager")
    
    def run_chain_of_agents(self, text : list, task_prompt, synthesis_prompt):
        """
        Execute the Chain of Agents framework on the input text.
        """
        # chunks = split_text(text, chunk_size)
        chunks = text
        prev_summary = ""
        worker_outputs = []
        input_prompts = []
        for worker_id, chunk in enumerate(chunks):
            worker_input_prompt, worker_output = self.worker_agent(prev_summary + chunk, task_prompt)
            # print(f"W {worker_id}:", worker_output)
            prev_summary = worker_output+"/n/n"
            worker_outputs.append(worker_output)
            input_prompts.append(worker_input_prompt)
        # worker_outputs = [worker_agent(client, chunk, task_prompt) for chunk in chunks]
        manager_input_prompt, final_output = self.manager_agent(worker_outputs, task_prompt, synthesis_prompt)
        input_prompts.append(manager_input_prompt)
        return input_prompts, final_output

    def run_chain_of_agents_with_reretrieval(
        self,
        initial_chunks: list,
        task_prompt,
        synthesis_prompt,
        seen_evidence_list,
    ):
        """
        Chain-of-Agents with cumulative re-retrieval:
        - Starts with initial chunks.
        - After each worker output, retrieves new evidence using (query + output).
        - Adds only new chunks (not already seen) to the list.
        - Each iteration processes one chunk in sequence.
        """
        prev_summary = ""
        worker_outputs = []
        input_prompts = []

        current_chunks = initial_chunks.copy()
        seen_evidence_titles = set(seen_evidence_list)  # to avoid repeats

        max_steps = len(current_chunks)*2

        step = 0
        while step < max_steps and step < len(current_chunks):
            chunk = current_chunks[step]

            # Combine context with summary
            context = prev_summary + "\n\n" + chunk

            # Step 1: Ask worker
            worker_input_prompt, worker_output = self.worker_agent(context, task_prompt)

            # Step 2: Store
            input_prompts.append(worker_input_prompt)
            worker_outputs.append(worker_output)

            # Step 3: Update summary
            prev_summary = worker_output + "\n\n"

            # Step 4: Re-retrieve
            new_evidence_list = search_similar(task_prompt + "\n\n" + prev_summary)

            new_evidence_filtered = []
            for e in new_evidence_list:
                if e["title"] not in seen_evidence_titles:
                    new_evidence_filtered.append(e)
                    seen_evidence_titles.add(e["title"])

            # Step 5: Convert to 2-chunk format, filter duplicates
            new_chunks = []
            for i in range(0, len(new_evidence_filtered), 2):
                if i + 1 < len(new_evidence_filtered):
                    combined = new_evidence_filtered[i]['text'] + "\n\n" + new_evidence_filtered[i + 1]['text']
                else:
                    combined = new_evidence_filtered[i]['text']

                new_chunks.append(combined)

            # Step 6: Append only new chunks
            current_chunks.extend(new_chunks)

            step += 1

        # Final synthesis
        manager_input_prompt, final_output = self.manager_agent(
            worker_outputs, task_prompt, synthesis_prompt
        )

        input_prompts.append(manager_input_prompt)
        return input_prompts, final_output

    """Code for generating chunks for worker LLMs"""

    def generate_context(self, evidence_list, database):
        context_chunks = []
        for evi_id, evidence in enumerate(evidence_list):
            document = database[evidence]
            whole_context = f"[Evidence {evi_id+1}] Title: {document['title']}\nBody: {document['body']}"

            if len(context_chunks) != 0 and len(whole_context.split()) + len(context_chunks[-1].split()) <= self.chunk_size:
                context_chunks[-1] += f"\n\n{whole_context}"
                continue

            while len(whole_context.split()) > self.chunk_size:
                whole_context_words = whole_context.split()

                # find full-stop to take the last complete sentence
                offset = -1
                for i in range(len(whole_context_words)-1, 0, -1):
                    if whole_context_words[i][-1] == '.':
                        break
                    else:
                        offset += 1
                chunk = ' '.join(whole_context_words[:self.chunk_size-offset])
                context_chunks.append(chunk)
                whole_context = f"[Evidence {evi_id+1}] Title: {document['title']}\nBody: " + ' '.join(whole_context_words[self.chunk_size-offset:])
            else:
                context_chunks.append(whole_context)

        return context_chunks

client = Together(api_key=os.getenv("TOGETHERAI_API_KEY")) # work


# Initialize the progress bar
total_queries = len(query_samples[args.query_start_id:])  # Total queries to process
pbar = tqdm.tqdm(total=total_queries, desc="Queries", dynamic_ncols=True, initial = args.query_start_id+1)

API_CALLS = []
save_results = []

if not os.path.exists(output_tsv_file):
  with open(output_tsv_file, 'w') as f:
    f.write("prompt\tprediction\tgold\n")

# Initialize the Chain of Agents
coa = ChainOfAgents(client, model=args.model, chunk_size=args.chunk_size)

for idx, query in enumerate(query_samples[args.query_start_id:]):
    try: 
        query_title = query['query']
    
        # evidence_list = [doc2id[evidence["title"]] for evidence in query["evidence_list"]]
        # context_list = generate_context(evidence_list, database, chunk_size=args.chunk_size)

        evidence_list = search_similar(query_title)
        titles = [doc["title"] for doc in evidence_list]

        # Combine every two pieces of evidence into one context
        context_list = []
        for i in range(0, len(evidence_list), 2):
            context_list.append(evidence_list[i]['text'] + "\n\n" + evidence_list[i+1]['text'])

    
        synthesis_prompt = (
            "Below is a question followed by some context from different sources. "
            "Please answer the question based on the context. The answer to the question is a word or entity. "
            "If the provided information is insufficient to answer the question, respond 'Insufficient Information'. "
            "Answer directly without explanation."
        )
    
        #input_prompts, result = coa.run_chain_of_agents(context_list, task_prompt=query_title, synthesis_prompt=synthesis_prompt)

        input_prompts, result = coa.run_chain_of_agents_with_reretrieval(context_list, task_prompt=query_title, synthesis_prompt=synthesis_prompt,seen_evidence_list=titles)

        # print(f"{input_prompts}\t{result}\t{query['answer']}\n")
        
        API_CALLS.append(len(context_list) + 1)
    
        with open(output_tsv_file, 'a') as f:
          f.write(f"{input_prompts}\t{result}\t{query['answer']}\n")
    
        save_results.append({"input_list": input_prompts, "prediction": result, "gold": query['answer']})
    
        # Update progress bar with API calls
        total_api_calls = sum(API_CALLS)
        pbar.set_description(f"Queries | API Calls: {total_api_calls}")
        pbar.refresh()
        pbar.update(1)  # Move progress bar forward
    except Exception as e:
        print("Error with query ID:", idx + args.query_start_id, "\t", query_title)
        print("\t",e)
        print(tb.format_exc())
        continue

pbar.close()  # Close the progress bar once done

with open(output_json_file, 'w') as f:
    json.dump(save_results, f)
