# VentureBoard AI - Backend Application

This directory contains the FastAPI-based backend services for VentureBoard AI. It orchestrates the LangGraph agents, runs semantic queries on ChromaDB/SimpleVectorStore, fetches search groundings, and builds diligence report PDFs.

## Installation & Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```
   Add your `GROQ_API_KEY` (and any other API keys required by the agents).

5. **Run the local development server:**
   ```bash
   python run.py
   # Or using uvicorn directly:
   uvicorn app.main:app --reload --port 8000
   ```

## Running Tests

Automated tests are located in the `tests/` directory. To run them:
```bash
pytest
```

## Railway Deployment Settings

When deploying this backend to Railway, configure the service settings as follows:
- **Root Directory:** `backend`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Environment Variables:** Define `GROQ_API_KEY`.
