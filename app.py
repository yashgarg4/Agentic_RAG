import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Define the path for the persistent vector store
PERSIST_DIRECTORY = 'db'

# Define the prompt template
# This guides the LLM on how to behave and what context to use.
prompt_template = """
You are a helpful and precise Partner Support Agent.
Use the following pieces of context to answer the user's question.
If you don't know the answer from the provided context, just say that you don't know, don't try to make up an answer.
Advise the user to contact support via email at support@example.com if the information is not in the context.

Context: {context}
Question: {question}

Helpful Answer:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def main():
    # Initialize the embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load the persisted vector store
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    # Create a retriever from the vector store
    retriever = db.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

    # Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    print("\nWelcome to the Partner Support Agent!")
    print("Ask a question about the documentation, or type 'exit' to quit.")

    # Interactive loop to take user questions
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            break
        if query.strip() == '':
            continue

        # Process the query through the QA chain
        result = qa_chain.invoke({"query": query})

        # Print the answer
        print("\nAnswer:")
        print(result["result"])

if __name__ == "__main__":
    main()
