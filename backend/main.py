from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

import uvicorn

from llama_index.llms.gemini import Gemini
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import JsonSerializer

# Import the agents
from agents.data_loading_agent import make_data_loading_agent
from agents.data_exploration_agent import make_data_exploration_agent
from agents.manager_agent import make_manager_agent
from agents.reporting_agent import make_reporting_agent
import pandas as pd
import io

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="DataHammer Analytics API")
llm = Gemini(
    model="models/gemini-2.5-pro",
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
async def analyze_data(
    file: UploadFile = File(...), user_instructions: str = Form(None)
):
    """
    Analyze data using the Manager Agent which coordinates between specialized agents.
    Loads and analyzes CSV files for data insights based on user instructions.
    """
    # Verify file is CSV
    if not file.filename.endswith(".csv"):
        return {
            "error": "Only CSV files are supported",
            "message": "Please upload a CSV file",
        }

    # Read the file content
    file_content = await file.read()

    # Create DataFrame from CSV content
    df = pd.read_csv(io.BytesIO(file_content))

    # Create all agents
    data_agent = make_data_loading_agent(llm=llm)
    exploration_agent = make_data_exploration_agent(llm=llm)
    reporting_agent = make_reporting_agent(llm=llm)
    manager_agent = make_manager_agent(llm=llm)

    # Set default analysis message if no user instructions provided
    analysis_message = "Analyze this CSV file and provide summary statistics"

    # Use user instructions if provided
    if user_instructions:
        analysis_message = f"Analyze this CSV file with the following instructions: {user_instructions}"

    # Initialize workflow with DataFrame in context and manager as root agent
    workflow = AgentWorkflow(
        agents=[data_agent, exploration_agent, reporting_agent, manager_agent],
        root_agent=exploration_agent.name,  # Set manager agent as the root
        initial_state={
            "Plan": "No plan made so far.",
            "DataFrame": df,
            "File Name": file.filename,
            "File Type": "CSV",
            "Observations": [],
            "User Question": (
                user_instructions
                if user_instructions
                else "Provide a comprehensive analysis of this dataset"
            ),
            "Analysis Status": "initialized",
        },
    )

    ctx = Context(workflow=workflow)

    # Run the CSV analysis workflow
    response = await workflow.run(user_msg=analysis_message, ctx=ctx)

    # # Get the final report and report sections from the state
    # final_report = ""
    # report_sections = []
    # # ctx_dict = ctx.to_dict(serializer=JsonSerializer())
    # state = ctx.get("state", {})

    # # Get final report or report sections from state using .get() method
    # print("State:", state)

    # final_report = state.get("final_report", "")
    # if not final_report:
    #     final_report = state.get("report_sections", "")
    # if not final_report:
    #     final_report = response

    return {"final_report": response}


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
