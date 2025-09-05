import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables from .env file
load_dotenv()

# Define the path for the persistent vector store
PERSIST_DIRECTORY = 'db'

def main():
    # Load documents from the 'data' folder for different file types
    pdf_loader = DirectoryLoader('data/', glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True)
    docx_loader = DirectoryLoader('data/', glob="**/*.docx", loader_cls=Docx2txtLoader, show_progress=True, use_multithreading=True)
    txt_loader = DirectoryLoader('data/', glob="**/*.txt", loader_cls=TextLoader, show_progress=True, use_multithreading=True)

    all_loaders = [pdf_loader, docx_loader, txt_loader]
    
    documents = []
    print("Loading documents...")
    for loader in all_loaders:
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading files with {loader.loader_cls.__name__}: {e}")
    print(f"Loaded {len(documents)} documents.")

    # Split the loaded documents into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    # Initialize the embedding model from Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("Initialized embeddings model.")

    # Create the vector store from the document chunks and embeddings
    # This will process the documents and store their vector representations
    # The 'persist_directory' argument ensures the data is saved to disk
    print("Creating vector store...")
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    # Persist the database to disk
    db.persist()
    db = None # Free up memory
    print("Vector store created and persisted.")

if __name__ == "__main__":
    main()
