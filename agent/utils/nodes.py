from langchain import hub
from langchain_openai import ChatOpenAI

from .tools import tools


def agent(state):
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

    model = ChatOpenAI(temperature=0.4, streaming=True, model="gpt-4o-mini")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response],
            "question": messages[-1].content
            }


def generate(state):
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
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.4, streaming=True)

    # Chain
    rag_chain = prompt | llm

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}
