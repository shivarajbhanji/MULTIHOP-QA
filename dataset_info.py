import json


with open('multihoprag/MultiHopRAG.json') as f:
    query_data = json.load(f)

query_token_count=0

print( query_data[0].keys())

max_len=0
for data_sample in query_data:
    query_token_count+=len(data_sample['query'].split())
    max_len=max(max_len,len(data_sample['query'].split()))

print(f"Query count: {len(query_data)}")
print(f"Query token count: {query_token_count}")
print(f"Avg. tokens per query: {int(query_token_count/len(query_data))}")
print(f"Max tokens in a query: {max_len}")
print()

with open('multihoprag/corpus.json') as f:
    corpus_data = json.load(f)

corpus_token_count=0
max_len=0

print( corpus_data[0].keys())

for document in corpus_data:
    corpus_token_count+=len(document['body'].split())
    max_len=max(max_len,len(document['body'].split()))

print(f"Corpus count: {len(corpus_data)}")
print(f"Corpus token count: {corpus_token_count}")
print(f"Avg. tokens per document: {int(corpus_token_count/len(corpus_data))}")
print(f"Max tokens in a document: {max_len}")