# DataHammer: Advanced Data Analytics Platform

A completely revamped Next.js application with Chakra UI and TypeScript for data analytics. This application allows users to upload spreadsheet files for data analysis, cleaning, visualization, and predictive modeling.

> **Note**: This PR proposes a complete overhaul of the DataHammer repository with a modern tech stack and enhanced features.

## Features

- File upload interface for spreadsheet data (CSV, XLS, XLSX)
- Natural language data processing instructions
- User-friendly interface for describing data operations
- Support for multiple data formats (CSV, JSON, XML, Plain Text)
- Data cleaning and preprocessing
- Exploratory data analysis
- Data visualization
- Insights generation
- Predictive modeling
- Local data processing
- GitHub Pages deployment

## Tech Stack

- **Frontend**: Next.js, TypeScript, Chakra UI, React Dropzone
- **Backend**: FastAPI, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Database**: Local storage (demo only)
- **Deployment**: GitHub Pages

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- Python (v3.8 or higher)
- pip (Python package manager)
- Docker and Docker Compose (for containerized deployment)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BainMcHale/DataHammer.git
   cd DataHammer
   ```

2. Install frontend dependencies:
   ```bash
   npm install
   ```

### Configuration

No additional configuration is needed for the demo version.

### Running the Application

#### Using Docker (Recommended)

1. Start the backend server using Docker:
   ```bash
   make backend
   ```

2. To stop the backend:
   ```bash
   make backend-stop
   ```

3. To view backend logs:
   ```bash
   make backend-logs
   ```

#### Manual Setup

1. Start the backend server:
   ```bash
   cd backend
   python main.py
   ```

2. In a separate terminal, start the frontend:
   ```bash
   npm run dev
   ```

3. Or use the provided script to run both:
   ```bash
   ./run.sh
   ```

4. Open your browser and navigate to `http://localhost:3000`

## Deployment

The application is configured for GitHub Pages deployment using GitHub Actions. When you push to the main branch, the workflow will automatically build and deploy the application.

To configure GitHub Pages:

1. Go to your repository settings
2. Navigate to Pages
3. Select the `gh-pages` branch as the source
4. No additional configuration is needed for the demo version
