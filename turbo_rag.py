import logging
import sys
import torch
import json
import time
from tabulate import tabulate
from qwen2 import Qwen2ModifiedForCausalLM
from transformers import AutoTokenizer

# Llama Index Related
from llama_index.core import Settings, load_index_from_storage, StorageContext, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

PREFIX = '''<|im_start|>system
You are an accurate and reliable AI assistant that can answer questions with the help of external documents. Please note that external documents may contain noisy information. If the information in the document contains the correct answer, you will give an accurate answer. If the information in the document does not contain the answer, you will generate ’I can not answer the question because of the insufficient information in documents.‘.<|im_end|><|im_start|>user\nDocs:'''

def stack_past_key_values(past_key_values_list):
    num_layers = len(past_key_values_list[0])
    batch_past_key_values = []
    for layer in range(num_layers):
        keys = torch.cat([past_key_values[layer][0] for past_key_values in past_key_values_list], dim=2)
        values = torch.cat([past_key_values[layer][1] for past_key_values in past_key_values_list], dim=2)
        batch_past_key_values.append((keys, values))
    return tuple(batch_past_key_values)

def qa_to_prompt(chunk_list, query):
    chunk_str = "".join(chunk_list)
    prompt = f'''{PREFIX}{chunk_str}\n\nQuestuin: {query}<|im_end|><|im_start|>assistant\n'''
    return prompt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "path_to_turborag_model"
model = Qwen2ModifiedForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
)
storage_context = StorageContext.from_defaults(persist_dir="doc_emb")
index = load_index_from_storage(storage_context)

retriever = index.as_retriever(similarity_top_k=20)

inputs_prefix = tokenizer([PREFIX], return_tensors="pt",padding=True)
outputs_prefix = model(
    inputs_prefix['input_ids'].to(device), 
    attention_mask = inputs_prefix['attention_mask'].to(device), 
    use_cache=True
)
prefix_kvcache = outputs_prefix.past_key_values

def query_with_kvcache(query_text, use_chunk_cache=True):
    query_bundle = QueryBundle(query_str=query_text)
    retrieved_nodes = retriever.retrieve(query_bundle)
    kvcache_list, chunk_list = [prefix_kvcache], []
    for node_with_score in retrieved_nodes:
        node = node_with_score.node  
        if use_chunk_cache:
            kvcache = torch.load(node.metadata["kvcache_file_path"])
            kvcache_list.append(kvcache)
        chunk_list.append(node.text)
        

    prompt = qa_to_prompt(chunk_list, query_text)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    past_kvcache = stack_past_key_values(kvcache_list) if use_chunk_cache else None
        
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=1,
            past_key_values=past_kvcache,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            eos_token_id=[151645,151643],
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    questions = []
    with open("./questions/query.jsonl") as file:
        for item in file:
            data = json.loads(item)
            questions.append(data["query"])

    # Test the average time taken for RAG with document chunk KV Cache
    start = time.perf_counter()
    for query in questions:
        answer = query_with_kvcache(query)
    end = time.perf_counter()
    use_time = end - start
    avg_time_with_cache = use_time / len(questions)
    
    # Test the average time taken for RAG without document chunk KV Cache
    start = time.perf_counter()
    for query in questions:
        answer = query_with_kvcache(query, use_chunk_cache=False)
    end = time.perf_counter()
    use_time_without_cache = end - start
    avg_time_without_cache = use_time_without_cache / len(questions)

    # Prepare data for tabular output
    results = [
        ["With KV Cache", f"{avg_time_with_cache:.6f} seconds"],
        ["Without KV Cache", f"{avg_time_without_cache:.6f} seconds"]
    ]

    # Print the results in a table format
    print(tabulate(results, headers=["Method", "Average Time"], tablefmt="grid"))