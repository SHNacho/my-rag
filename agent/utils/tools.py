import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


##### Retriever #####
files = ["data/my-cv.md", "data/my-hobbies.md"]

@st.cache_resource(ttl="1h")
def configure_retriever(files):
    # Read documents
    docs = []
    for file_path in files:
        loader = UnstructuredMarkdownLoader(file_path)
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        persist_directory="chroma",
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings()
    )

    # Define retriever
    retriever = vectorstore.as_retriever()

    return retriever


retriever_tool = create_retriever_tool(
    configure_retriever(files),
    "retrieve_ignacio_information",
    "Search and return information about Ignacio",
)

tools = [retriever_tool]

