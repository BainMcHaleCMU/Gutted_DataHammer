<img width="818" alt="Book Example" src="https://github.com/user-attachments/assets/6f60ca95-0381-4804-b849-eadb35842adf" />
 #Gutted_DataHammer: Advanced Data Analytics Platform

A modern Next.js application with Chakra UI and TypeScript for data analytics. This application allows users to upload spreadsheet files for data analysis, cleaning, visualization, and predictive modeling with enhanced AI-powered capabilities.

This repository features a robust backend with improved modeling agents for more accurate data analysis and predictions.

## Features

- File upload interface for spreadsheet data (CSV, XLS, XLSX)
- Natural language data processing instructions
- User-friendly interface for describing data operations
- Support for multiple data formats (CSV, JSON, XML, Plain Text)
- Data cleaning and preprocessing
- Exploratory data analysis
- Data visualization
- Insights generation
- Advanced predictive modeling with:
  - Dynamic model selection based on data characteristics
  - Intelligent feature selection
  - Support for regression, classification, clustering, and time series analysis
  - Comprehensive error handling and validation
- Local data processing
- GitHub Pages deployment

## Tech Stack

- **Frontend**: 
  - Next.js 15.3.0
  - React 19.0.0
  - TypeScript 5
  - Chakra UI 2.8.2
  - React Dropzone 14.3.8
  - Axios for API communication
  - Jest and React Testing Library for testing

- **Backend**: 
  - FastAPI 0.115.12
  - Uvicorn 0.34.0
  - Pandas for data manipulation
  - LlamaIndex with Gemini for AI-powered analysis
  - Python 3.8+

- **Database**: Local storage (demo only)
- **Deployment**: GitHub Pages and Docker for containerization

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- Python (v3.8 or higher)
- pip (Python package manager)
- Docker and Docker Compose (for containerized deployment)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BainMcHaleCMU/Gutted_DataHammer.git
   cd Gutted_DataHammer
   ```

2. Install frontend dependencies:
   ```bash
   npm install
   ```

3. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   cd ..
   ```

### Configuration

1. For the backend to use Gemini AI capabilities, you'll need to set up environment variables:
   - Create a `.env` file in the `backend` directory
   - Add your Gemini API key: `GEMINI_API_KEY=your_api_key_here`

2. No additional configuration is needed for the frontend demo version.

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

## Project Structure

- `/src` - Frontend Next.js application
- `/backend` - FastAPI backend server
  - `/agents` - AI modeling agents for data analysis
  - `main.py` - Main FastAPI application
- `/dummyData` - Sample data for testing
- `/docs` - Documentation files
- `/public` - Static assets for the frontend

## Key Improvements

This repository includes significant improvements to the ModelingTaskAgent and ModelingAgent components:

1. **Comprehensive Error Handling**: Added validation and detailed error messages at multiple levels
2. **Dynamic Model Selection**: Intelligent model selection based on data characteristics
3. **Target Variable Detection**: Logic to determine target variables from multiple sources
4. **Feature Selection**: Intelligent feature selection based on dataset findings
5. **Dataset Validation**: Validation for different dataset types (tabular, time series, text)
6. **Support for Different Data Types**: Support for tabular, time series, and text data
7. **Improved Logging**: Detailed logging at each processing step

## Deployment

The application is configured for GitHub Pages deployment using GitHub Actions. When you push to the main branch, the workflow will automatically build and deploy the application.

To configure GitHub Pages:

1. Go to your repository settings
2. Navigate to Pages
3. Select the `gh-pages` branch as the source

For backend deployment, Docker containerization is recommended using the provided Dockerfile and docker-compose.yml.

## Contributing

Contributions to Gutted_DataHammer are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request

Please make sure to update tests as appropriate and follow the existing code style.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
