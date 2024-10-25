# ----- #
import os
import streamlit as st

from dotenv import load_dotenv

from agent.RAG import RagAgent 
from utils.logger import logger
# ----- #

# load env
if not load_dotenv():
    load_dotenv("/etc/secrets/env/.env")

# page configuration
st.set_page_config(page_title="Ignacio's Chatbot", layout="centered", page_icon="🤖")

# Set up session state
if 'messages' not in st.session_state:
    st.session_state.messages = []


##### Streamlit Chat Interface #####
def display_chat(graph):
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

        logger.log_text(user_input) # log user input
        st.session_state.messages.append(("user", user_input))
        st.chat_message("user").write(user_input)

        inputs = {"messages": [("user", user_input)]}
        config = {"configurable": {"thread_id": "1"}}

        for output in graph.stream(inputs, config=config):
            for key, value in output.items():

                message = value["messages"][0]

                if key == "generate": #or key == "agent":
                    st.session_state.messages.append(("bot", message.content))
                    st.chat_message("assistant").write(message.content)
                elif key == "agent" and message.content != '':
                    st.session_state.messages.append(("bot", message.content))
                    st.chat_message("assistant").write(message.content)

                logger.log_text(st.session_state.messages[-1][1])

# main
st.title("Ignacio's Chatbot")
rag = RagAgent()
rag.init_retriever() 
graph = rag.get_graph() # get the agent graph
display_chat(graph) # render the chat
