#!/bin/bash

# Check if Docker is available
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo "Docker detected, starting backend with Docker..."
    make backend
    USING_DOCKER=true
else
    echo "Docker not detected, starting backend directly..."
    # Start the FastAPI backend
    cd backend
    python main.py &
    BACKEND_PID=$!
    USING_DOCKER=false
fi

# Start the Next.js frontend
cd "$(dirname "$0")"  # Return to the root directory
npm run dev -- -p 12000 --hostname 0.0.0.0 &
FRONTEND_PID=$!

# Handle cleanup on exit
cleanup() {
    echo "Cleaning up..."
    kill $FRONTEND_PID
    
    if [ "$USING_DOCKER" = false ] && [ -n "$BACKEND_PID" ]; then
        kill $BACKEND_PID
    else
        make backend-stop
    fi
    
    exit
}

trap cleanup INT TERM EXIT

# Keep the script running
wait