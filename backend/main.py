from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import uvicorn


app = FastAPI(title="DataHammer Analytics API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze")
async def analyze_data(file: UploadFile = File(...)):
    """
    Analyze data using the traditional approach.
    Returns fixed sample responses for now.
    """
    
    return {
        "message": "Traditional data analysis completed",
        "data": {
            "sample_response": "This is a sample response from traditional analysis.",
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
