import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from dotenv import load_dotenv
import os

load_dotenv()
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

@tool
def get_weather(location: str):
   
    return f"The weather in {location} is currently 72°F and sunny."

@tool
def calculate_sum(a: int, b: int):

    return str(a + b)

tools = [get_weather, calculate_sum]
tool_node = ToolNode(tools)


llm = ChatGroq(
    model_name=os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile"),
    temperature=0.0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    """
    This node simply invokes the ChatGroq model with the current messages.
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def run_langgraph_tools_demo():
    print("Building LangGraph Pipeline with Tools...")

    graph_builder = StateGraph(State)

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)
    
   
    graph_builder.add_edge(START, "chatbot")
    

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    
    graph_builder.add_edge("tools", "chatbot")
    app = graph_builder.compile()
    print("Pipeline built successfully!\n")
    print("Welcome to the LangGraph Groq Chatbot with Tools!")
    print("Try asking for the weather or doing some math.")
    print("Type 'quit' or 'exit' to exit.\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            if not user_input.strip():
                continue

            events = app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                stream_mode="updates"
            )
            
            for event in events:
               
                for node_name, state_update in event.items():

                    latest_message = state_update["messages"][-1]
                    
                    if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                        for tc in latest_message.tool_calls:
                            print(f"-> [Agent is calling tool '{tc['name']}' with args {tc['args']}]")
                    
                    elif isinstance(latest_message, ToolMessage):
                        print(f"<- [Tool returned: {latest_message.content}]")
                        
                    elif latest_message.type == "ai" and latest_message.content:
                        print(f"Groq Bot: {latest_message.content}")
                    
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_langgraph_tools_demo()