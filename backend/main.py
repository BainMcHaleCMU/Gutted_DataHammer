from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

import uvicorn

from llama_index.llms.gemini import Gemini
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool

# Import the new DataLoadingAgent
from agents.data_loading_agent import make_data_loading_agent

# Import the new DataExplorationAgent
from agents.data_exploration_agent import make_data_exploration_agent
import pandas as pd
import io

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
    Analyze data using the DataLoadingAgent.
    Loads and analyzes CSV files for data insights.
    """
    # Verify file is CSV
    if not file.filename.endswith(".csv"):
        return {
            "error": "Only CSV files are supported",
            "message": "Please upload a CSV file",
        }

    # Read the file content
    file_content = await file.read()

    # Convert file content to DataFrame

    # Create DataFrame from CSV content
    df = pd.read_csv(io.BytesIO(file_content))

    # Create a data loading agent specialized for CSV processing
    data_agent = make_data_loading_agent(llm=llm)

    # Create a data exploration agent for data analysis
    exploration_agent = make_data_exploration_agent(llm=llm)

    # Initialize workflow with DataFrame in context
    workflow = AgentWorkflow(
        agents=[data_agent, exploration_agent],
        root_agent=data_agent.name,
        initial_state={
            "Plan": "Process and analyze CSV data",
            "DataFrame": df,
            "File Name": file.filename,
            "File Type": "CSV",
            "Observations": [],
        },
    )

    # Run the CSV analysis workflow
    response = await workflow.run(
        user_msg="Analyze this CSV file and provide summary statistics"
    )

    return {
        "message": "CSV analysis completed",
        "data": response,
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
