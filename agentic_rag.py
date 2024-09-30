import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

#### Streamlit Setup #####

# Streamlit page configuration
st.set_page_config(page_title="LangGraph Chatbot", layout="centered")

# Set up session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

##### Retriever #####
files = ["data/my-cv.md", "data/my-hobbies.md"]

@st.cache_resource(ttl="1h")
def configure_retriever(files):
    print("Embedding")
    # Read documents
    docs = []
    for file_path in files:
        loader = UnstructuredMarkdownLoader(file_path)
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
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

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    configure_retriever(files),
    "retrieve_ignacio_information",
    "Search and return information about Ignacio",
)

tools = [retriever_tool]

##### Agent State #####

from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, BaseMessage

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

##### Edges #####

from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field


from langgraph.prebuilt import tools_condition

### Edges

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        return "generate"

    else:
        return "rewrite"


### Nodes


def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)

    # Chain
    rag_chain = prompt | llm

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

##### Graph #####

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile
graph = workflow.compile()


#inputs = {
#    "messages": [
#        ("user", "Hi!")
#    ]
#}
#for output in graph.stream(inputs):
#    for key, value in output.items():
#        pprint.pprint(f"Output from node '{key}':")
#        pprint.pprint("---")
#        pprint.pprint(value, indent=2, width=80, depth=None)
#    pprint.pprint("\n---\n")

##### Streamlit Chat Interface #####
def display_chat():
    # Display previous messages
    for role, content in st.session_state.messages:
        if role == "user":
            st.write(f"**User:** {content}")
        else:
            st.write(f"**Bot:** {content}")

    # Input box for user query
    user_input = st.chat_input("Ask Ignacio a question:")
    if user_input:
        st.session_state.messages.append(("user", user_input))
        st.write(f"**User:** {user_input}")

        inputs = {"messages": [("user", user_input)]}
        print("-------New------")
        for output in graph.stream(inputs):
            for key, value in output.items():
                print("Key: ",key)
                print("Value: ",value)
                print('----------------')
                message = value["messages"][0]
                if key == "generate": #or key == "agent":
                    st.session_state.messages.append(("bot", message.content))
                    st.write(f"**Bot:** {message.content}")
                elif key == "agent" and message.content != '':
                    st.session_state.messages.append(("bot", message.content))
                    st.write(f"**Bot:** {message.content}")


# Call the display_chat function to render the chat
st.title("LangGraph Chatbot")
display_chat()
