import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct, CollectionStatus, UpdateStatus
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client.http import models
from typing import List

import os
import sys
sys.path.insert(0, os.getcwd())

# preprocessing
from langchain.document_loaders import PyPDFLoader
from ingest.ingest_pdf import load_pdf_folder, chunk_pdf_docs, docs_to_add

# embedding
from embed.embed_pdf import get_HFembedding

# embedding model
from langchain.embeddings import OpenAIEmbeddings

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.vectorstores import Qdrant
import uuid

import re

class QdrantCollection:
    def __init__(
        self,
        folder_path: str,
        collection_name: str,
        host: str = "http://localhost/6333",  #http://10.0.0.52:6333/dashboard#/collections (nidhi-rag.demo.lan link)
        port: int = 6333,
        #  db_path: str = "/home/nidhi/code/qdrant-rag/qdrant-db",
        embeddingfunction=get_HFembedding,
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        embedding_chunk_size=1000,
        vector_size: int = 768,
        vector_distance=Distance.COSINE,  
    ):
        self.client = QdrantClient(url=host)
        #    port=port,
        #    path=db_path)

        self.collection_name = collection_name

        self.embeddingmodel = embeddingfunction(embedding_model)

        self.folder_path = folder_path

        try:
            collection_info = self.client.get_collection(collection_name=collection_name)
        except Exception as e:
            print("Collection does not exist, creating collection now")
            self.set_up_collection(collection_name, vector_size, vector_distance)

    # create new collection using folder contents.
    def set_up_collection(self, collection_name: str, vector_size: int, vector_distance: str):
        # create collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=vector_distance, on_disk=True),
            optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
            on_disk_payload=True,
        )

        collection_info = self.client.get_collection(collection_name=collection_name)

        # embed folder contents
        docs = load_pdf_folder(folder_path=self.folder_path)

        if len(docs) == 0:
            print("No documents found in folder")
            return

        if len(docs) != 0:
            # if folder is not empty, chunk the documents.
            preprocessed_docs = chunk_pdf_docs(docs, chunk_size=400, chunk_overlap=100)

            # embed document chunks and put them in new collection
            collection_instance = Qdrant(self.client, self.collection_name, self.embeddingmodel)
            collection_instance.add_documents(preprocessed_docs)

        message = f"New collection '{self.collection_name}' created from PDF files located at '{self.folder_path}'."
        print(message)
        return collection_instance

    def upsert_collection(self, docs_to_be_added: list):
        collection_info = self.client.get_collection(collection_name=self.collection_name)

        # load pdf files
        # pdf filepath ---> langchain doc

        if len(docs_to_be_added) == 0:
            print("Collection is up-to-date")
            print("No documents to be added in the collection.")

            return

        # if collection needs to be updated:
        doc_list = []
        for doc in docs_to_be_added:
            # load pdf file
            loader = PyPDFLoader(doc)
            doc_list.append(loader.load())  # OUTPUT = list of list of dict.

            # flatten list of list for chunking.
            documents_with_metadata = [
                Document for another_list in doc_list for Document in another_list  # langchain Document object
            ]

        # chunk pdf files
        # Data formet of output = Langchain Document object
        chunked_docs = chunk_pdf_docs(documents_with_metadata, chunk_size=400, chunk_overlap=100)

        # upsert
        points = []
        for doc in chunked_docs:
            page_content = doc.page_content
            metadata = doc.metadata

            text_vector = self.embeddingmodel.embed_query(page_content)
            text_id = str(uuid.uuid4())

            # create point
            point = PointStruct(
                id=text_id, vector=text_vector, payload={"page_content": page_content, "metadata": metadata}
            )

            points.append(point)

        operation_info = self.client.upsert(collection_name=self.collection_name, wait=True, points=points)
        if operation_info.status == UpdateStatus.COMPLETED:
            print("Data inserted successfully!")
        else:
            print("Failed to insert data")

    @classmethod
    def fetch_collection(cls, collection_name: str):
        client = QdrantClient("localhost", port=6333)

        collections_str = str(client.get_collections())

        # Using regular expression to extract collection names
        collection_names = re.findall(r"name='([^']+)'", collections_str)

        if (
            collection_name in collection_names
        ):  # don't use str- use regex. Will get errors with full and partial spellings.
            collection_instance = Qdrant(client, collection_name, embeddings=OpenAIEmbeddings())
            return collection_instance

    @classmethod
    def check_collection_exists(cls, collection_name: str):
        exists = False

        client = QdrantClient("localhost", port=6333)

        collections_str = str(client.get_collections())
        collection_names = re.findall(r"name='([^']+)'", collections_str)
        print(collection_names)

        for name in collection_names:
            if name == collection_name:
                exists = True
                
        print(f"Collection {collection_name} exists!")
        return exists

    @classmethod
    def fetch_or_create_collection(cls, self, collection_name="Document Store", folder_path="/root/data/"):
        if QdrantCollection.check_collection_exists(collection_name):
            collection = QdrantCollection.fetch_collection(collection_name)

        else:
            collection = QdrantCollection(collection_name, folder_path)

        return collection

    # TODO:
    def delete_docs():
        pass

    def delete_collection():
        pass

    def collection_config():
        pass


# first instantiate collection.

# test = QdrantCollection(collection_name="Document Store", folder_path="/root/data/")
# test.upsert_collection([])

# collection_name = "Document Store",
# folder_path = "/root/data/"

# collection = QdrantCollection(collection_name,
#                                   folder_path,
#                                   )

# test = QdrantCollection(collection_name= "Document Store", folder_path="/root/data/")
# test.upsert_collection([])

#create new collection
lasik = QdrantCollection(collection_name= "Document Store", folder_path = "/root/mistral-demo/data")
