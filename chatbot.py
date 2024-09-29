import json
import os
import tempfile
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import SentenceTransformerEmbeddings 
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import prompts

st.set_page_config(page_title="Myself", page_icon="🦉")
st.title("Chat with me")


@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        if temp_filepath.endswith('.pdf'):
            loader = PyPDFLoader(temp_filepath)
        elif temp_filepath.endswith('.md'):
            loader = UnstructuredMarkdownLoader(temp_filepath)
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

    return retriever

class LogsCallback(BaseCallbackHandler):

    def on_retriever_start(self, serialized, query, **kwargs):
        print("Question: ", query)
        print("Retrieving documents ...")

    def on_retriever_end(self, documents, **kwargs):
        retrieved_docs = {}
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            retrieved_docs[idx] = {
                "source": source,
                "content": doc.page_content
            }
        #print("Retrieved documents:")
        #print(json.dumps(retrieved_docs, indent=4))

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        print("Prompts: ", prompts)



class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


uploaded_files = st.sidebar.file_uploader(
    label="Upload data files", type=["pdf", "md"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

retriever = configure_retriever(uploaded_files)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory(key="chat_history")
# memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
llm = ChatOllama(
    model="mistral", temperature=0
)

contextualize_q_prompt = ChatPromptTemplate(
    [
        ("system", prompts.contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate(
    [
        ("system", prompts.qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
qa_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)
    msgs.add_user_message(user_query)

    logs_callback = LogsCallback()
    response = rag_chain.invoke(
        {"input": user_query, "chat_history": msgs}, 
        {"callbacks": [logs_callback]}
    )
    st.chat_message("assistant").write(response["answer"])
    msgs.add_ai_message(response["answer"])
