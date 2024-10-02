from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from .utils import (
    tools,
    nodes,
)

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: BaseMessage

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", nodes.agent)  # agent
retrieve = ToolNode([tools.retriever_tool])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node(
    "generate", nodes.generate
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


