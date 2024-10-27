# ----- #
import streamlit as st
import random

from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph

from agent.RAG import RagAgent 
from utils.logger import logger
# ----- #

# load env
if not load_dotenv():
    load_dotenv("/etc/secrets/env/.env")

# page configuration
st.set_page_config(page_title="Ignacio's Chatbot", layout="centered", page_icon="🤖")
if 'thread' not in st.session_state:
    st.session_state.thread = random.randint(1, 4999999)
    print("New user with thread ", st.session_state.thread)

# Set up session state
if 'messages' not in st.session_state:
    st.session_state.messages = []


##### Streamlit Chat Interface #####
def display_chat(graph: CompiledStateGraph):
    avatars = {"user": "user", "bot": "assistant"}

    if len(st.session_state.messages) == 0:
        st.session_state.messages.append(
            ("bot", "Hi! I'm Ignacio's assistant, how can I help you?")
        )
    
    # Display previous messages
    for role, content in st.session_state.messages:
        st.chat_message(avatars[role]).write(content)

    # Input box for user query
    if user_input := st.chat_input("Ask anything about Ignacio:"):

        st.session_state.messages.append(("user", user_input))
        st.chat_message("user").write(user_input)

        inputs = {"messages": [("user", user_input)]}
        config = {"configurable": {"thread_id": st.session_state.thread}}

        print(inputs)
        output = graph.invoke(inputs, config=config)
        message = output['messages'][-1]
        st.session_state.messages.append(("bot", message.content))
        st.chat_message("assistant").write(message.content)

        logger.log_text(user_input) # log user input
        logger.log_text(message.content) # log chatbot output

# main
st.title("Ignacio's Chatbot")
rag = RagAgent()
rag.init_retriever() 
graph = rag.get_graph() # get the agent graph
display_chat(graph) # render the chat
