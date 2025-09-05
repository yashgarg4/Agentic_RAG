# Agentic RAG based IntelliAgent

This project implements a Retrieval-Augmented Generation (RAG) system to create an "IntelliAgent." This agent can answer questions based on a private collection of documents. It provides multiple interfaces, from a simple command-line tool to a sophisticated multi-agent web application.

## Features

- **Document Ingestion**: Processes and indexes various document formats (`.pdf`, `.docx`, `.txt`) from a local directory.
- **Vector Storage**: Uses `ChromaDB` to create and persist a vector store of document embeddings.
- **LLM Integration**: Leverages Google's `Gemini` models for generating embeddings and conversational responses.
- **Multiple Interfaces**:
  - **CLI App (`app.py`)**: A basic command-line interface for stateless Q&A.
  - **Streamlit App (`streamlit_app.py`)**: A web-based chat interface that remembers conversation history.
  - **Agentic Streamlit App (`streamlit_agentic.py`)**: An advanced web interface powered by `CrewAI` that uses multiple agents (Researcher and Answerer) for more robust, multi-step retrieval and response synthesis.
- **Evaluation**: Includes a script (`evaluate.py`) for basic keyword-based evaluation of the RAG pipeline's performance.

## Project Structure

```
.
├── data/                  # Place your source documents here
├── db/                    # Persistent ChromaDB vector store (created after ingestion)
├── app.py                 # Simple command-line Q&A interface
├── streamlit_app.py       # Streamlit app with chat history
├── streamlit_agentic.py   # Advanced Streamlit app with CrewAI agents
├── ingest.py              # Script to process documents and build the vector store
├── evaluate.py            # Script to run evaluation tests on the agent
├── .env                   # Environment variables (API keys)
└── requirements.txt       # Python dependencies
```

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    A `requirements.txt` file should be created containing the necessary packages. You can create one with the key dependencies listed below.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a file named `.env` in the root directory and add your Google API key:
    ```
    GOOGLE_API_KEY="your_google_api_key_here"
    ```
    You can obtain a key from Google AI Studio.

## Usage

### Step 1: Add Your Documents

Place all your support documents (`.pdf`, `.docx`) inside the `data/` directory. You can create subdirectories if you wish.

### Step 2: Ingest the Data

Run the ingestion script to process the documents, create embeddings, and build the vector store. This only needs to be done once, or whenever you add/update documents.

```bash
python ingest.py
```

This will create a `db/` directory containing the Chroma vector store.

### Step 3: Run an Application

You can interact with the agent using one of the following applications.

#### Option A: Command-Line Interface

For simple, single-turn questions.

```bash
python app.py
```

#### Option B: Streamlit Web App

For a chat-like experience that remembers conversation history.

```bash
streamlit run streamlit_app.py
```

#### Option C: Agentic Streamlit Web App (Advanced)

For a more powerful, multi-step reasoning process using CrewAI agents.

```bash
streamlit run streamlit_agentic.py
```

### Step 4 (Optional): Evaluate the Agent

You can run a simple evaluation to check if the agent's responses contain expected keywords for a given set of questions.

1.  **Customize the evaluation set:** Open `evaluate.py` and modify the `EVALUATION_SET` list with questions and expected keywords relevant to your documents.
2.  **Run the script:**
    ```bash
    python evaluate.py
    ```

## Key Dependencies

Your `requirements.txt` should include at least the following packages:

```
langchain
langchain-community
langchain-google-genai
google-generativeai
chromadb
streamlit
crewai
python-dotenv
pypdf
docx2txt
```

