import os
import streamlit as st

from dotenv import load_dotenv

from agent.RAG import RagAgent 


if not load_dotenv():
    load_dotenv('/etc/secrets/env/.env')

# Streamlit page configuration
st.set_page_config(page_title="LangGraph Chatbot", layout="centered")

# Set up session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

files = []
data_path = "data"
if not os.path.isdir(data_path):
    data_path = "/etc/data"
for file in os.listdir(data_path):
    files.append(os.path.join(data_path, file))

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
    user_input = st.chat_input("Ask anything about Ignacio:")
    if user_input:
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


# Call the display_chat function to render the chat
st.title("LangGraph Chatbot")
rag = RagAgent()
rag.init_retriever(files)
graph = rag.get_graph()
display_chat(graph)
