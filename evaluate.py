import asyncio
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

PERSIST_DIRECTORY = 'db'

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

condense_question_prompt_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_prompt_template)

def load_chain():
    """Loads the conversational chain."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    # Use temperature 0 for deterministic evaluation
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.0)
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key='answer'
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return chain

# --- Evaluation Framework ---

# Customize this with questions and keywords from YOUR documents.
EVALUATION_SET = [
    {
        "question": "What is the process for getting an API key?",
        "keywords": ["dashboard", "generate", "api key", "developer"],
    },
    {
        "question": "How do I handle authentication errors?",
        "keywords": ["401", "unauthorized", "token", "header"],
    },
    {
        "question": "What are the rate limits for the production environment?",
        "keywords": ["rate limit", "production", "requests per second"],
    }
]

def main():
    # Workaround for asyncio issue on Windows
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    qa_chain = load_chain()
    total_passed = 0

    print("--- Starting Evaluation ---")
    for i, item in enumerate(EVALUATION_SET):
        question = item["question"]
        expected_keywords = item["keywords"]
        print(f"\n--- Test Case {i+1}: {question} ---")

        result = qa_chain.invoke(question)
        answer = result["answer"]
        passed = all(keyword.lower() in answer.lower() for keyword in expected_keywords)

        print(f"Agent's Answer: {answer}")
        print(f"Result: {'PASSED ✅' if passed else 'FAILED ❌'}")
        if passed:
            total_passed += 1
        else:
            missing = [k for k in expected_keywords if k.lower() not in answer.lower()]
            print(f"Reason: Missing keywords -> {missing}")

    print("\n--- Evaluation Summary ---")
    print(f"Passed: {total_passed}/{len(EVALUATION_SET)} ({total_passed/len(EVALUATION_SET):.0%})")
    print("--------------------------")

if __name__ == "__main__":
    main()


