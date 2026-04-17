from ingest.ingest_pdf import load_pdf_folder, chunk_pdf_docs

import torch
import argparse
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.vectorstores import Qdrant

# from langchain.embeddings import HuggingFaceBgeEmbeddings

# from langchain.llms import OpenAI
# from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
import os

# # #OpenAI API key
# os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

# for getting pdf filenames for caching.
import glob


# def get_OpenAIembedding(model_name: str = "text-embedding-ada-002", embedding_chunk_size=1000):
#     embeddings = OpenAIEmbeddings(
#         model=model_name,
#         chunk_size=embedding_chunk_size,
#     )
#     return embeddings

def get_HFembedding(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    
    model_kwargs = {'device':'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
    model_name=model_name,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
    )    
    
    return embeddings
    
    
    


def clear_cache():
    qdrant_client = QdrantClient(url="http://localhost:6333")
    collection = qdrant_client.get_collection("vector_db")
    collection.clear_cache()


def embed_prompts():
    pass


def update_db():
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection_name", type=str, default="vector_db")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-large-en")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--folder_path", type=str, default="/root/data/")

    args = parser.parse_args()
    create_db(args.folder_path, args.collection_name, args.device)


if __name__ == "__main__":
    main()
