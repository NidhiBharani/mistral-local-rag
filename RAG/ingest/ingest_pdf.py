from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
import glob

# from llm import get_embedding


def get_folder_contents(folder_path):
    # get pdf files from it.
    pdf_files = glob.glob(folder_path + "/*.pdf")

    return pdf_files

    # TODO: later add other text types later.


# def embed_pdf(pdf_file):
#     embeddingmodel = get_embedding()


# if more docs have been added to folder.
def docs_to_add(folder_path, old_folder_contents: list):
    current_contents = get_folder_contents(folder_path)
    docs = [x for x in current_contents if x not in old_folder_contents]

    return docs


# if docs have been removed from folder.
def docs_to_remove(folder_path, old_folder_contents: list):
    current_contents = get_folder_contents(folder_path)
    docs = [x for x in old_folder_contents if x in current_contents]

    return docs


def load_pdf_folder(folder_path):
    loader = PyPDFDirectoryLoader(folder_path)
    docs = loader.load()
    # print(docs[0])
    return docs


def chunk_pdf_docs(preprocessed_pdfs: list, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    texts = text_splitter.split_documents(preprocessed_pdfs)

    return texts


# load_pdf_folder("/root/data/")


# folder_path = "/root/data/"
# docs = load_pdf_folder(folder_path)
# texts = chunk_pdf_docs(docs, chunk_size=3000, chunk_overlap=100)
# print("First text: ",texts[0])
