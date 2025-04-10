# import groq
from together import Together
import tqdm
import os
import json
import numpy as np
import argparse


parser = argparse.ArgumentParser(description="Process file paths for MultiHopRAG.")

parser.add_argument("-r","--root", type=str, required=True, help="Root directory path.")
parser.add_argument("-o", "--output_file_name", type=str, required=True, help="Output file name.")
parser.add_argument("-c", "--chunk_size", type=int, default=1500, help="Chunk size for splitting text.")
parser.add_argument("-m", "--model", type=str, default="git checkout -b your-branch-name", help="Model name.")

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
  query["gold_eveidence"] = gold_retrievals


""" Code for Chain of Agents"""

def gpt3_completion(client, prompt, role, model=args.model, temperature=0.7):

    if role == "worker":
      system_prompt = "You are an AI assistant that performs multi-hop reasoning based on multiple pieces of evidence. Summarize the context you have that might hold the answer."
    elif role == "manager":
      system_prompt = "You are an AI assistant that performs multi-hop reasoning based on multiple pieces of evidence. Provide one word answer."

    response = client.chat.completions.create(
    model=model,
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content

def split_text(documents_list, doc2id, chunk_size):
    """
    Split the input text into chunks of specified size.
    """
    chunks = []
    for document in documents_list:
        text = document["text"]
        doc_id = doc2id[document["title"]]
        words = text.split()

    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def worker_agent(client, chunk, task_prompt):
    """
    Process a chunk of text with a specific task prompt.
    """

    prompt = f"Question: {task_prompt}\n\n" + \
              f"Context: {chunk}" + \
              f"Answer:"
    # prompt = f"{task_prompt}\n\n{chunk}"
    return prompt, gpt3_completion(client, prompt, role="worker")

def manager_agent(client, worker_outputs, task_prompt, synthesis_prompt):
    """
    Synthesize the outputs from worker agents into a final response.
    """
    combined_output = "\n\n".join(worker_outputs)

    prompt = f"{synthesis_prompt}\n\n" + \
              f"Question: {task_prompt}\n\n" + \
              f"Context: {combined_output}\n" + \
              f"Answer:"
    # prompt = f"{synthesis_prompt}\n\n{combined_output}"
    return prompt, gpt3_completion(client, prompt, role="manager")

def chain_of_agents(client, text : list, task_prompt, synthesis_prompt):
    """
    Execute the Chain of Agents framework on the input text.
    """
    # chunks = split_text(text, chunk_size)
    chunks = text
    prev_summary = ""
    worker_outputs = []
    input_prompts = []
    for worker_id, chunk in enumerate(chunks):
        worker_input_prompt, worker_output = worker_agent(client, prev_summary + chunk, task_prompt)
        # print(f"W {worker_id}:", worker_output)
        prev_summary = worker_output+"/n/n"
        worker_outputs.append(worker_output)
        input_prompts.append(worker_input_prompt)
    # worker_outputs = [worker_agent(client, chunk, task_prompt) for chunk in chunks]
    manager_input_prompt, final_output = manager_agent(client, worker_outputs, task_prompt, synthesis_prompt)
    input_prompts.append(manager_input_prompt)
    return input_prompts, final_output


"""Code for generating chunks for worker LLMs"""

def generate_context(evidence_list, database, chunk_size):
  context_chunks = []
  for evi_id, evidence in enumerate(evidence_list):
    document = database[evidence]
    whole_context = f"[Evidence {evi_id+1}] Title: {document['title']}\nBody: {document['body']}"

    if len(context_chunks) != 0 and len(whole_context.split()) + len(context_chunks[-1].split()) <= chunk_size:
      context_chunks[-1] += f"\n\n{whole_context}"
      continue

    while len(whole_context.split()) > chunk_size:
      whole_context_words = whole_context.split()

      # find full-stop to take the last complete sentence
      offset = -1
      for i in range(len(whole_context_words)-1, 0, -1):
        if whole_context_words[i][-1] == '.':
          break
        else:
          offset += 1
      chunk = ' '.join(whole_context_words[:chunk_size-offset])
      context_chunks.append(chunk)
      whole_context = f"[Evidence {evi_id+1}] Title: {document['title']}\nBody: " + ' '.join(whole_context_words[chunk_size-offset:])
    else:
      context_chunks.append(whole_context)

  return context_chunks

#x = generate_context([0,1], database, chunk_size=1500)
#len(x),x


API_CALLS = []
for query in query_samples[args.query_start_id:]:
 query_title = query['query']

 evidence_list = [doc2id[evidence["title"]] for evidence in query["evidence_list"]]
 context_list = generate_context(evidence_list, database, chunk_size=args.chunk_size)

 API_CALLS.append(len(context_list) + 1)

print("Total API Calls needed: ", sum(API_CALLS))
print("Average number of API Calls needed:", np.mean(API_CALLS))
print("List of `query_id:API` Calls...")
print({i+args.query_start_id:api for i, api in enumerate(API_CALLS)})

"""Run the CoA over all the queries"""

client = Together(api_key="a67211c50a58f3a06a06caf34e8115e4b213ef2aaf29690d8a9789a9ea10268") # work

q_start_id = args.query_start_id

# Initialize the progress bar
total_queries = len(query_samples[q_start_id:])  # Total queries to process
pbar = tqdm.tqdm(total=total_queries, desc="Queries", dynamic_ncols=True, initial = q_start_id+1)

API_CALLS = []
save_results = []

if not os.path.exists(output_tsv_file):
  with open(output_tsv_file, 'w') as f:
    f.write("prompt\tprediction\tgold\n")

sub_list = [query_samples[i] for i in [17, 31, 80, 121, 212, 246, 585, 589, 614, 694, 1023, 1424, 1429, 1440, 2417, 2502]]

for idx, query in enumerate(sub_list):
    try: 
        query_title = query['query']
    
        evidence_list = [doc2id[evidence["title"]] for evidence in query["evidence_list"]]
        context_list = generate_context(evidence_list, database, chunk_size=args.chunk_size)
    
        synthesis_prompt = (
            "Below is a question followed by some context from different sources. "
            "Please answer the question based on the context. The answer to the question is a word or entity. "
            "If the provided information is insufficient to answer the question, respond 'Insufficient Information'. "
            "Answer directly without explanation."
        )
    
        input_prompts, result = chain_of_agents(client, context_list, task_prompt=query_title, synthesis_prompt=synthesis_prompt)
        
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
        print("Error with query ID:", idx + q_start_id, "\t", query_title)
        print("\t",e)
        continue

pbar.close()  # Close the progress bar once done

with open(output_json_file, 'w') as f:
    json.dump(save_results, f)
