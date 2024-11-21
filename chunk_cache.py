import torch
from transformers import AutoTokenizer
from qwen2 import Qwen2ModifiedForCausalLM
from typing import List, Optional
from dataclasses import dataclass
from tqdm import tqdm
import os

# LlamaIndex related
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    PromptHelper,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import SimpleVectorStore

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "path_to_turborag_model" 
model = Qwen2ModifiedForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

splitter = TokenTextSplitter(
    tokenizer=tokenizer.encode,
    chunk_size=512,
    chunk_overlap=10
)

output_path = "chunk_kvcache"
if not os.path.exists(output_path):
    os.makedirs(output_path)

def process_chunk(chunk_text, chunk_id):
    chunk_text = "<|doc_start|>" + chunk_text + "<|doc_end|>"
    inputs = tokenizer(chunk_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    
    past_key_values = outputs.past_key_values
    
    kvcache_file_path = f'{output_path}/kvcache_chunk_{chunk_id}.pt'
    torch.save(past_key_values, kvcache_file_path)
    
    node = TextNode(
        text=chunk_text,
        id_=f"chunk_{chunk_id}",
        metadata={
            "kvcache_file_path": kvcache_file_path
        }
    )
    
    return node

class KVCachedNodeParser(SimpleNodeParser):
    def get_nodes_from_documents(
        self,
        documents: List[Document],
        **kwargs,
    ) -> List[BaseNode]:
        nodes = []
        for doc_id, document in tqdm(enumerate(documents)):
            doc_text = document.get_content()
            chunk_texts = splitter.split_text(doc_text)
            
            for chunk_id, chunk_text in enumerate(chunk_texts):
                node = process_chunk(chunk_text, f"{doc_id}_{chunk_id}")
                nodes.append(node)
        return nodes

embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
)

vector_store = SimpleVectorStore()
node_parser = KVCachedNodeParser()
documents = SimpleDirectoryReader('documents').load_data()
nodes = node_parser.get_nodes_from_documents(documents)

index = VectorStoreIndex(
    nodes=nodes,
    embed_model=embed_model,
    vector_store=vector_store,
)

index.storage_context.persist(persist_dir='doc_emb')