from embed.vectorstore import QdrantCollection

# database related
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant

# # llm related
# from llm import get_llm

# retriever related
from langchain.retrievers.multi_query import MultiQueryRetriever

# logging
import logging

# llm = get_llm()


# def get_multiqueryretriever(collection_name="Document Store", llm=llm):
#     collection = QdrantCollection.fetch_or_create_collection(collection_name)

#     retriever = MultiQueryRetriever.from_llm(retriever=collection.as_retriever(), llm=llm)

#     return retriever

#uncomment to test
# retriever = get_multiqueryretriever()

# question
# question = "Why is the sky blue?"

#uncomment to test
# Set logging for the queries
# logging.basicConfig()
# logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# use retriever to get related chunks for the queries.
# unique_docs = retriever.get_relevant_documents(query=question)
# print(len(unique_docs))
# print(unique_docs)
