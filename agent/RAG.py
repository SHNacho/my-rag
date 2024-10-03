import streamlit as st

from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing import Literal, Annotated, Sequence
from typing_extensions import TypedDict


class RagAgent():


    class AgentState(TypedDict):
        # The add_messages function defines how an update should be processed
        # Default is to replace. add_messages says "append"
        messages: Annotated[Sequence[BaseMessage], add_messages]
        question: BaseMessage

    def __init__(self, openai_model: str = "gpt-4o-mini") -> None:
        self.tools = []
        self.retriever = None
        self.retriever_tool = None
        self.model = openai_model
        self.graph = None

    ### TOOLS ###
    @st.cache_resource(ttl="1h")
    def init_retriever(_self, files):
        # Read documents
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
    
        vectorstore = Chroma.from_documents(
            documents=splits,
            persist_directory="chroma",
            collection_name="rag-chroma",
            embedding=OpenAIEmbeddings()
        )
    
        # Define retriever
        _self.retriever = vectorstore.as_retriever()
        _self.retriever_tool = create_retriever_tool(
            _self.retriever,
            "retrieve_ignacio_information",
            "Search and return information about Ignacio",
        )
        _self.tools.append(_self.retriever_tool)

    ### NODES ###
    def agent(self, state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.
    
        Args:
            state (messages): The current state
    
        Returns:
            dict: The updated state with the agent response appended to messages
        """
        system = {
            "role": "system",
            "content": (
                "You are Ignacio's assistant. You only answer question about him."
            )
        }
        messages = state["messages"]
        if len(messages) == 1 :
            messages.insert(0, system)
    
        model = ChatOpenAI(temperature=0.4, streaming=True, model=self.model)
        model = model.bind_tools(self.tools)
        response = model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response],
                "question": messages[-1].content
                }
    
    
    def generate(self, state):
        """
        Generate answer
    
        Args:
            state (messages): The current state
    
        Returns:
             dict: The updated state with re-phrased question
        """
        messages = state["messages"]
        question = state["question"]
        last_message = messages[-1]
        print(messages)
    
        docs = last_message.content
    
        # Prompt
        prompt = hub.pull("rlm/rag-prompt")
    
        # LLM
        llm = ChatOpenAI(model_name=self.model, temperature=0.4, streaming=True)
    
        # Chain
        rag_chain = prompt | llm
    
        # Run
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}

    ### EDGES ###
    def grade_documents(self, state) -> Literal["generate", "rewrite"]:
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
        model = ChatOpenAI(temperature=0, model=self.model, streaming=True)
    
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
    
        question = messages[1].content
        docs = last_message.content
    
        scored_result = chain.invoke({"question": question, "context": docs})
    
        score = scored_result.binary_score
    
        if score == "yes":
            return "generate"
    
        else:
            return "rewrite"

    @st.cache_resource(ttl="1h")
    def get_graph(_self):
        assert _self.retriever, "Retriever has not been initialized"
        # Define a new graph
        workflow = StateGraph(_self.AgentState)
        
        # Define the nodes we will cycle between
        workflow.add_node("agent", _self.agent)  # agent
        retrieve = ToolNode([_self.retriever_tool])
        workflow.add_node("retrieve", retrieve)  # retrieval
        workflow.add_node(
            "generate", _self.generate
        )  # Generating a response
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
        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory)
        _self.graph = graph
        return _self.graph
        


        
