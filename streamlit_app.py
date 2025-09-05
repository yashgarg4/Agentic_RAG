import streamlit as st
from dotenv import load_dotenv
import asyncio
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Define the path for the persistent vector store
PERSIST_DIRECTORY = 'db'

# Define the prompt template
qa_template = """
You are a helpful and precise Partner Support Agent.
Use the following pieces of context to answer the user's question.
If you don't know the answer from the provided context, just say that you don't know, don't try to make up an answer.
Advise the user to contact support via email at support@example.com if the information is not in the context.

Context: {context}
Question: {question}

Helpful Answer:
"""
QA_PROMPT = PromptTemplate(
    template=qa_template, input_variables=["context", "question"]
)

# --- New Prompt for Condensing Questions ---
condense_question_prompt_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_prompt_template)

# Use Streamlit's caching to load the model and retriever only once.
@st.cache_resource
def load_chain():
    """
    Loads the conversational chain, which includes the LLM, vector store, and retriever.
    This function is cached to avoid reloading on every user interaction.
    """
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

    # --- New Memory Object ---
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key='answer'
    )

    # --- Use ConversationalRetrievalChain instead of RetrievalQA ---
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return chain

def main():
    # This is a workaround for a known issue with google-generativeai and Streamlit
    # The gRPC client used by the Google library requires an asyncio event loop to be
    # running in the current thread, which Streamlit does not provide by default.
    try:
        asyncio.get_running_loop()
    except RuntimeError:  # 'RuntimeError: There is no current event loop...'
        # If there is no current event loop, create one and set it for the current thread.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    st.title("Partner Support Agent")
    st.write("Ask a question about our documentation, and the agent will find the answer for you.")

    # --- New: Initialize chat history in session state ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load the conversational chain
    chain = load_chain()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if query := st.chat_input("Your question:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Finding the answer..."):
            # Get the result from the chain
            result = chain.invoke(query)
            answer = result["answer"]
            source_documents = result["source_documents"]

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("Sources"):
                    for doc in source_documents:
                        st.write(f"**Source:** {os.path.basename(doc.metadata['source'])}")
                        st.write(doc.page_content)
                        st.write("---")

if __name__ == "__main__":
    main()
