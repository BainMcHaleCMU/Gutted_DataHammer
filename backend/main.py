from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

import uvicorn

from llama_index.llms.gemini import Gemini
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent

# from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="DataHammer Analytics API")
llm = Gemini(
    model="models/gemini-1.5-flash",
    api_key=os.environ.get("GOOGLE_API_KEY"),
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two integers and returns the result integer"""
    return a - b


def divide(a: int, b: int) -> int:
    """Divides two integers and returns the result integer"""
    return a / b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
divide_tool = FunctionTool.from_defaults(fn=divide)


@app.post("/analyze")
async def analyze_data(file: UploadFile = File(...)):
    """
    Analyze data using the traditional approach.
    Returns fixed sample responses for now.
    """
    agent = FunctionAgent(
        tools=[multiply_tool, add_tool, subtract_tool, divide_tool],
        llm=llm,
        system_prompt="You are a helpful assistant that can search the web for information.",
    )
    workflow = AgentWorkflow(agents=[agent])

    response = await workflow.run(user_msg="What is 47 * 32?")
    return {
        "message": "Traditional data analysis completed",
        "data": {
            "sample_response": response,
        },
    }


@app.get("/")
async def root():
    return {
        "message": "DataHammer Analytics API is running",
        "endpoints": {
            "/analyze": "Traditional data analysis",
            "/agent-swarm/analyze": "AI Agent Swarm data analysis",
            "/agent-swarm/agents": "List available agents",
        },
    }


if __name__ == "__main__":
    # Run the server with CORS enabled
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
