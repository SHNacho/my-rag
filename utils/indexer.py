import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# load env
if not load_dotenv():
    load_dotenv("/etc/secrets/env/.env")


def index_documents(files: list[str]):
    """
    Index the documents into a Pinecone vectorial database

    Args:
        files: list of path to the files to index
    """
    docs = []
    for file_path in files:
        loader = UnstructuredMarkdownLoader(file_path)
        docs.extend(loader.load())
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, 
        chunk_overlap=0
    )
    splits = text_splitter.split_documents(docs)
    
    pc = Pinecone()
    index = pc.Index("my-rag")
    vectorstore = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings())
    vectorstore.add_documents(
        documents=splits
    )


# data path (for indexing)
data_path = "data" # local data dir
if not os.path.isdir(data_path):
    data_path = "/etc/data" # gcp data dir

# create list with data files
files = []
for file in os.listdir(data_path):
    files.append(os.path.join(data_path, file))

index_documents(files)
