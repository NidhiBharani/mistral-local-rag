from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
# from langchain.embeddings import OpenAIEmbeddings
import os
import argparse
import torch
# # Initialize OpenAI API key
# os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline


def create_retrieval_qa(
    collection_name,
    llm_model,
    embedding_model,  # retriever)
):
    # Initialize OpenAI embeddings
    embeddings = get_embedding()

    # Initialize Qdrant client and database
    url = "http://localhost:6333"
    qdrantclient = QdrantClient(url=url, prefer_grpc=False)
    db = Qdrant(qdrantclient, collection_name=collection_name, embeddings=embeddings)  # "Document Store"

    # Setup retrieval from the provided Qdrant database
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    print("Initialize OpenAI model")
    # Create a chain for question answering
    qa = RetrievalQA.from_chain_type(
        llm=llm_model, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    return qa


def call_qa_chain(qa_chain, query):
    result = qa_chain({"query": query})
    return result


# def get_llm(model_name="gpt-3.5-turbo-instruct", max_tokens=400, temperature=0.9):
    
#     if model_name=="gpt-3.5-turbo-instruct":
        
#         from langchain.llms import OpenAI
#         llm = OpenAI(model_name=model_name, max_tokens=max_tokens, temperature=temperature)
#     return llm

def get_mistral_model(model_name='mistralai/Mistral-7B-Instruct-v0.1'):
    
    #config
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
									bnb_4bit_quant_type="nf4", 
									bnb_4bit_use_double_quant=True,
                                    bnb_4bit_compute_dtype=torch.float16
         )

    #load model
    model = AutoModelForCausalLM.from_pretrained(
                                                model_name,
                                                quantization_config=bnb_config, ##load_in_4bit=True - no need for this now.
                                                device_map = 'cuda', #trust_remote_code=True - no need.
    )
    print(model_name)
    return model

# def get_embedding(embeddings=OpenAIEmbeddings()):
#     embeddings = embeddings
#     return embeddings


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddingmodel", type=str, default="OpenAIEmbeddings()")
    parser.add_argument(
        "--LLMmodel", type=str, default='OpenAI(model="gpt-3.5-turbo-instruct", temperature=2, max_tokens=4086)'
    )
    parser.add_argument("--query", type=str, default="What is cancer?")

    args = parser.parse_args()

    # Perform a query
    query = "what is the total number of instances of metastasis?"

    qa_chain = create_retrieval_qa(llm_model=args.LLMmodel, embedding_model=args.embeddingmodel)

    # Perform a query
    result = call_qa_chain(qa_chain, query=args.query)
    print(result)
    print(result["result"])


if __name__ == "__main__":
    main()
