import streamlit as st

from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, RemoveMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pinecone import Pinecone
from pydantic import BaseModel, Field
from typing import Literal, Annotated, Sequence
from typing_extensions import TypedDict

from utils.logger import logger

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
        self.llm = ChatOpenAI(temperature=0.4, streaming=True, model=self.model)
        self.graph = None

    ### TOOLS ###
    @st.cache_resource(ttl="1h")
    def init_retriever(_self):
        # Read documents
        pc = Pinecone()
        index = pc.Index("my-rag")
        vectorstore = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings())

        # Define retriever
        _self.retriever = vectorstore.as_retriever(k=2)
        _self.retriever_tool = create_retriever_tool(
            _self.retriever,
            "retrieve_ignacio_information",
            "Search and return information about Ignacio",
        )
        _self.tools.append(_self.retriever_tool)

    ### NODES ###
    def filter(self, state):
        """
        Filter the messages deleting tool calls and documents retrieved.

        Args:
            state (messages): The current state
        
        Returns:
            dict: The updated state
        """
        print("===Filter===")
        remove_ids = [m.id for m in state["messages"] if isinstance(m, ToolMessage) or not m.content]
        delete_messages = [RemoveMessage(id=id) for id in remove_ids]
        return {"messages": delete_messages}
    
    def contextualize_question(self, state):
        """
        Give context to the question for a better understanding

        Args:
            state (messages): The current state

        Returns:
            dict: The contextualized question
        """
        print("===Contextualize===")
        prompt = (
            "Given the above conversation, reformulate the last question for a "
            "better understanding without the context if possible. Otherwise "
            "paraphrase the last user input:"
        )
        messages = state["messages"] + [HumanMessage(content=prompt)]

        contextualized_question = self.llm.invoke(messages)

        return {
            "question": contextualized_question
        }

    def agent(self, state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.
    
        Args:
            state (messages): The current state
    
        Returns:
            dict: The updated state with the agent response appended to messages
        """
        print("===Agent===")
        system = SystemMessage(content="You are Ignacio's assistant. You only answer question about him.")
        question = state["messages"][-1]
        messages = state["messages"]

        feed_messages = [system] + messages[:-1] + [question]
    
        model = ChatOpenAI(temperature=0.4, streaming=True, model=self.model)
        model = model.bind_tools(self.tools)
        response = model.invoke(feed_messages)
        # We return a list, because this will get added to the existing list
        return {
            "question": [question],
            "messages": [response],
        }
    
    
    def generate(self, state):
        """
        Generate answer with the documents context.
    
        Args:
            state (messages): The current state
    
        Returns:
             dict: The updated state with the response
        """
        print("===Generate===")
        messages = state["messages"]
        question = state["question"]
        last_message = messages[-1]
    
        docs = last_message.content
        logger.log_text("Documents: " + docs)
    
        # Prompt
        prompt = hub.pull("rlm/rag-prompt")
    
        # Chain
        rag_chain = prompt | self.llm
    
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
    def get_graph(_self) -> CompiledStateGraph:
        assert _self.retriever, "Retriever has not been initialized"
        # Define a new graph
        workflow = StateGraph(_self.AgentState)
        
        # Define the nodes we will cycle between
        workflow.add_node("filter", _self.filter) # filter state
        # workflow.add_node("contextualize", _self.contextualize_question)
        workflow.add_node("agent", _self.agent)  # agent
        retrieve = ToolNode([_self.retriever_tool])
        workflow.add_node("retrieve", retrieve)  # retrieval
        workflow.add_node("generate", _self.generate)  # Generating a response

        # Define edges
        workflow.add_edge(START, "filter")
        #workflow.add_edge("filter", "contextualize")
        workflow.add_edge("filter", "agent")
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
        


        
