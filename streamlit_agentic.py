import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from typing import Any, List, Dict

# LangChain / Vectorstore
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# CrewAI
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import PrivateAttr

# ---------- Setup ----------
load_dotenv()
PERSIST_DIRECTORY = "db"

# ---------- RAG Tool ----------
class RAGRetrieverTool(BaseTool):
    name: str = "rag_retriever"
    description: str = (
        "Use this to search the internal partner docs. "
        "Returns the most relevant chunks with their sources."
    )

    _retriever: Any = PrivateAttr()

    def __init__(self, retriever):
        super().__init__()
        self._retriever = retriever

    def _run(self, query: str) -> str:
        docs = self._retriever.get_relevant_documents(query)
        if not docs:
            return "NO_RESULTS"
        lines = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "unknown")
            lines.append(f"[{i}] SOURCE: {src}\n{d.page_content}")
        return "\n\n".join(lines)

# ---------- Build Vectorstore & Retriever ----------
@st.cache_resource
def build_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    return retriever

# ---------- LLM ----------
@st.cache_resource
def build_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.2,
        api_key=os.getenv("GEMINI_API_KEY")
    )

# ---------- Crew ----------
def build_crew(chat_history: List[Dict[str, str]]):
    retriever = build_retriever()
    llm = build_llm()
    rag_tool = RAGRetrieverTool(retriever)

    researcher = Agent(
        name="Researcher",
        role="Query Planner & Evidence Gatherer",
        goal="Plan sub-queries and call retriever multiple times until enough context is gathered.",
        backstory="Skilled at breaking down complex questions and retrieving evidence iteratively.",
        tools=[rag_tool],
        llm="gemini/gemini-1.5-flash-latest",
        allow_delegation=False,
    )

    answerer = Agent(
        name="Answerer",
        role="Synthesis & Verification",
        goal="Craft a precise, helpful answer with citations. If insufficient info, say so.",
        backstory="Careful and precise, avoids hallucination, always cites sources.",
        tools=[rag_tool],
        llm="gemini/gemini-1.5-flash-latest",
        allow_delegation=True,
    )

    plan_and_retrieve = Task(
        description=(
            "User question:\n{question}\n\nChat history:\n{chat_history}\n\n"
            "Steps:\n1) Plan sub-queries.\n2) Use rag_retriever multiple times.\n"
            "3) Summarize findings as evidence digest."
        ),
        agent=researcher,
        expected_output=(
            "A concise evidence digest with bullet points of key facts "
            "and short notes on remaining gaps. Do not give final answer."
        ),
    )

    synthesize_and_verify = Task(
        description=(
            "Using the evidence digest, craft a final answer. "
            "Include [source: filename] citations. If missing info, say so clearly."
        ),
        agent=answerer,
        context=[plan_and_retrieve],
        expected_output=(
            "A clear, structured final answer to the user question, "
            "with inline [source: filename] citations. If insufficient evidence, "
            "say 'I don’t know' and advise contacting support@example.com."
        ),
    )

    crew = Crew(
        agents=[researcher, answerer],
        tasks=[plan_and_retrieve, synthesize_and_verify],
        process=Process.sequential,
    )
    inputs = {
        "question": st.session_state.get("latest_question", ""),
        "chat_history": "\n".join([f"{m['role']}: {m['content']}" for m in chat_history]),
    }
    return crew, inputs

# ---------- Streamlit UI ----------
def ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

def main():
    ensure_event_loop()
    st.title("IntelliAgent")
    # st.caption("CrewAI + Chroma + Gemini, multi-step retrieval and synthesis.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! Ask me about the docs."}
        ]

    # Render past messages
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Handle user input
    user_msg = st.chat_input("Your question…")
    if not user_msg:
        return

    st.session_state.messages.append({"role": "user", "content": user_msg})
    st.session_state.latest_question = user_msg

    with st.chat_message("assistant"):
        with st.spinner("Thinking with multi-step retrieval…"):
            crew, inputs = build_crew(st.session_state.messages)
            result = crew.kickoff(inputs=inputs)
            final_answer = result.raw if hasattr(result, "raw") else str(result)
            st.markdown(final_answer)

    # Save message
    st.session_state.messages.append({"role": "assistant", "content": final_answer})

if __name__ == "__main__":
    main()
