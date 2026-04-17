# Streamlit
import os
import sys
sys.path.append(os.getcwd())

import streamlit as st
import time

#for loading retriever
# Database related imports
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from embed.vectorstore import QdrantCollection
from embed.embed_pdf import get_HFembedding

# ingest
from ingest.ingest_pdf import docs_to_add, get_folder_contents

# prompts
from prompts import basic_prompt

# llm
from transformers import AutoTokenizer,pipeline
from llm import get_mistral_model

#langchain related
import langchain
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

#reordering
# from langchain.document_transformers import LongContextReorder

# for exception handling
from urllib.error import URLError
from collections import defaultdict

import torch

# set defaults for session_state
defaults = {
    "folder_path": "/root/data/",
    "collection_name": "Document Store",
    "llm": "",
    "embedding": "",
    "context": "",
    "chain": None,
    "messages": [],
    "old_folder_contents": [],
    "collection": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# get collection name and folder path from st.session_state and pass it to update_collection
# I could write it directly but cannot pass st.session_state.collection_name as argument.
collection_name = defaults["collection_name"]
folder_path = defaults["folder_path"]


@st.cache_resource
def update_collection(
    old_folder_contents: list,
    collection_name=collection_name,
    folder_path="folder_path",
):
    docs = docs_to_add(folder_path, old_folder_contents)

    # initialize collection or get it if it already exists.
    collection = QdrantCollection(collection_name=collection_name, 
                                  folder_path=folder_path)

    # upsert collection
    collection.upsert_collection(docs)

    st.session_state[old_folder_contents] = get_folder_contents(folder_path)

    print("Collection updated.")
    print("session state updated- old_folder_contents")


# replace this with something advanced later
def llm_cache():
    from langchain.cache import InMemoryCache

    langchain.llm_cache = InMemoryCache()


# def clear_cache():
#     # didn't test it but should work
#     #  langchain.llm_cache.clear()

#     if not st.session_state["llm"]:
#         st.warning("Could not find llm to clear cache of")
#     llm = st.session_state["llm"]
#     llm_string = llm._get_llm_string()
#     langchain.llm_cache.clear(llm_string=llm_string)

def clear_cache():
    torch.cuda.empty_cache()

def reset_app():
    st.session_state["query"] = ""
    st.session_state["messages"].clear()
    st.session_state["collection"] = None


 #prepare llm 
model_name='mistralai/Mistral-7B-Instruct-v0.1'
#tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

#pipeline
text_generation_pipeline = pipeline(
                                    model=get_mistral_model(),
                                    tokenizer = tokenizer,
                                    task="text-generation",
                                    do_sample = True,
                                    temperature=0.2,
                                    repetition_penalty=1.1,
                                    return_full_text=True,
                                    max_new_tokens=4096
)

#llm in correct RAG pipeline format.
llm = HuggingFacePipeline(model_id = model_name,
                            pipeline=text_generation_pipeline)

try:
    llm_cache()
    prompt = basic_prompt()

    with st.sidebar:
        st.write("## LLM Settings")
        ##st.write("### Prompt") TODO make possible to change prompt
        st.write("Change these before you run the app!")
        # max_tokens = st.slider("Number of Tokens", 100, 13072, 8000, key="max_tokens")

        st.write("## Retrieval Settings")
        st.write("Feel free to change these anytime")
        num_context_docs = st.slider("Number of Context Documents", 2, 20, 8, key="num_context_docs")
        distance_threshold = st.slider("Distance Threshold", 0.1, 0.8, 0.6, key="distance_threshold", step=0.1)

        st.write("## App Settings")
        st.button("Clear Chat", key="clear_chat", on_click=lambda: st.session_state["messages"].clear())
        st.button("Clear Cache", key="clear_cache", on_click=clear_cache)
        st.button("New Conversation", key="reset", on_click=reset_app)

    st.title("RAG")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Ask me something!**")
        question = st.text_input("Question", key="question")
        
    with col2:
        st.image("logo.png", width = 256)

    if st.button("Chat!"):
        with st.spinner("Loading information from data source...."):
            time.sleep(3)

            # collection
            client = QdrantClient("localhost", port=6333)
            collection_instance = Qdrant(
                                        client, 
                                        collection_name=collection_name, 
                                        embeddings=get_HFembedding()
                                        )
            
            st.session_state["collection"] = collection_instance
            
           
            
            #load llm to streamlit 
            if st.session_state["llm"] is not None:
                tokens = st.session_state["max_tokens"]
                print("Tokens ready!")
                st.session_state["llm"] = llm
                print(st.session_state["llm"])

            if st.session_state["embedding"]:
                st.session_state["embedding"] = get_HFembedding()

            try:
                llm = st.session_state["llm"]
                embedding = st.session_state["embedding"]
                
                #retriever
                fetch_k = min(num_context_docs, 30) #redundant- slider range between 2-20.
                k = min(fetch_k, 6)
                print(k)
                retriever = collection_instance.as_retriever(
                    search_type="mmr", 
                    search_kwargs={"k": k, 
                                   "fetch_k": fetch_k,
                                   'score_threshold': distance_threshold
                                   }
            )

                
               
                # reorder = LongContextReorder()
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff", 
                    retriever=retriever, 
                    return_source_documents=True,
                )                      
                

            except AttributeError:
                st.info("Please ask a question. I cannot read minds. Yet :)")
                st.stop()

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

                    print(st.session_state.messages)  # so far info about the llm model
                    print("**************")

            with st.chat_message("Answer"):
                message_placeholder = st.empty()
                st.session_state["context"], st.session_state["response"] = [], ""
                chain = st.session_state["chain"]

                result = chain({"query": question})

                st.markdown(result["result"])
                st.session_state["context"], st.session_state["response"] = result["source_documents"], result["result"]

                print("------------")
                print(st.session_state["context"])
                print("-------------")

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
        """
        % e.reason
    )
