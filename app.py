import os
import openai
import chainlit as cl

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
# from llama_index.core.callbacks import CallbackManager
# from llama_index.core.service_context import ServiceContext
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentInput, AgentStream, ToolCallResult, AgentOutput
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent, ReActAgent

from shoe_recommender_engine import ShoeRecommenderEngine
from shoe_retrieval import ShoeRetrieval


file_path = 'data/2025 HOOP SHEET PROCESSED.csv'
llm_kwargs = {'model': 'gpt-4o-mini', 'request_timeout': 360.0, 'temperature': 0.1}


@cl.on_chat_start
async def start():
    cl.run_sync(
        cl.Message(author="Assistant", content="Hello! I'm a basketball shoe AI assistant. How may I help you?").send()
    )
    
    Settings.llm = OpenAI(**llm_kwargs)
    Settings.embed_model = HuggingFaceEmbedding(model_name="jinaai/jina-embeddings-v3", trust_remote_code=True)
    
    shoe_recommender_tool = ShoeRecommenderEngine(file_path, llm_kwargs).as_tool()
    shoe_retrieval_tool = ShoeRetrieval(file_path).as_tool()

    shoe_recommender_agent = FunctionAgent(name = "shoe_recommender_agent",
                                    description = "An agent capable of recommending basketball shoes based on user's preference using recommender tool.",
                                    system_prompt = "You are a helpful assistant that recommends basketball shoes with clear explanationbased on user's preference using recommender tool. Handoff to shoe_retrieval_agent when specific shoe name(s) is given.",
                                    tools = [shoe_recommender_tool],
                                    can_handoff_to = ["shoe_retrieval_agent"]
                                    )
    shoe_retrieval_agent = FunctionAgent(name = "shoe_retrieval_agent",
                                    description = "An agent capable of retrieving specs given basketball shoe name(s) and discuss the performance of the shoe(s). "\
                                        "If multiple shoes are given, the agent should discuss the performance of each shoe one by one and compare them. This agent should be used **ONLY** when the shoe name(s) is given by the user.",
                                    system_prompt = "You are a helpful assistant that retrieves specs given basketball shoe name(s) and discuss the performance of the shoe(s) "\
                                        "If multiple shoes are given, the agent should discuss the performance of each shoe one by one and compare them. Use global_median_values as a reference of the average shoes' performance" \
                                        "Handoff to shoe_recommender_agent when the user asks for recommendations without specific shoe name(s).",
                                    tools = [shoe_retrieval_tool],
                                    can_handoff_to = ["shoe_recommender_agent"]
                                    )
    workflow = AgentWorkflow(agents = [shoe_recommender_agent, shoe_retrieval_agent], root_agent = 'shoe_recommender_agent',initial_state = {"recommendations": []})

    context = Context(workflow)
    cl.user_session.set("workflow", workflow)
    cl.user_session.set("context", context)

    


@cl.on_message
async def main(message: cl.Message):
    workflow = cl.user_session.get("workflow")
    context = cl.user_session.get("context")

    msg = cl.Message(content="", author="Assistant")

    handler = workflow.run(message.content, context = context)

  
    
    async for event in handler.stream_events():
        if isinstance(event, AgentInput):
            print(f"{event.current_agent_name} INPUT:{event.input}")
        if isinstance(event, AgentOutput) and event.response.content:
            print(f"{event.current_agent_name} OUTPUT: {event.response.content}")
        if isinstance(event, ToolCallResult):
            print(f"Tool called: {event.tool_name} -> {event.tool_output}")
        if isinstance(event, AgentStream):
            print(f"{event.delta}", end = "")
            await msg.stream_token(event.delta)
        
    await msg.send()