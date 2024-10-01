import streamlit as st
from agent.rag_agent import graph

##### Streamlit Chat Interface #####
def display_chat():
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
        for output in graph.stream(inputs):
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
display_chat()
