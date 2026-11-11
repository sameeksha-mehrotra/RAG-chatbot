from dotenv import load_dotenv
import os
import time
import requests
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from typing import Optional, List

load_dotenv()


def get_secret(name):
    """Read a key from .env locally, or from Streamlit Secrets when deployed."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return None


# Pinecone reads its key from the environment, so copy it over from Secrets if needed
if get_secret("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = get_secret("PINECONE_API_KEY")


class GroqLLM(LLM):
    api_key: str
    model_id: str = "openai/gpt-oss-20b"

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.3,
            "reasoning_effort": "low"
        }
        url = "https://api.groq.com/openai/v1/chat/completions"
        response = None
        for attempt in range(6):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
            except requests.exceptions.RequestException:  # dropped connection or timeout
                time.sleep(5)
                continue
            if response.status_code == 429:  # rate limited
                time.sleep(10)
                continue
            break
        if response is None:
            return "Sorry, I couldn't reach the model. Please try again."
        if response.status_code != 200:
            return f"API Error: {response.text}"
        return response.json()["choices"][0]["message"]["content"]


PROMPT = PromptTemplate.from_template("""
Answer the question using ONLY the context below. The answer is often a specific
number, name, or list, so read every passage carefully before saying it is missing.
Do not add facts, examples, or models that are not in the context.
If the context truly doesn't contain the answer, say "I don't know."

Context: {context}

Question: {question}

Answer:""")


@st.cache_resource
def load_components():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = PineconeVectorStore(
        index_name="rag-chatbot",
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )
    llm = GroqLLM(api_key=get_secret("GROQ_API_KEY"))
    return retriever, llm


def page_of(doc):
    # PyPDFLoader counts pages from 0; show the page number a PDF viewer shows
    page = doc.metadata.get("page")
    return int(page) + 1 if page is not None else None


def answer_with_sources(question):
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    answer = llm.invoke(PROMPT.format(context=context, question=question))
    sources = [{"page": page_of(d), "text": d.page_content} for d in docs]
    return answer, sources


def show_sources(answer, sources):
    if not sources or "i don't know" in answer.lower():
        return
    pages = sorted({s["page"] for s in sources if s["page"] is not None})
    st.caption("Sources: " + ", ".join(f"p. {p}" for p in pages))
    with st.expander("See the passages this answer was based on"):
        for s in sorted(sources, key=lambda s: s["page"] or 0):
            st.markdown(f"**Page {s['page']}**")
            st.write(s["text"])
            st.divider()


# UI
st.title("RAG Document Chatbot for LLMs in Finance")
st.caption(
    "Ask questions about *A Comprehensive Review of Generative AI in Finance* "
    "(FinTech, 2024). Every answer cites the pages it came from."
)

retriever, llm = load_components()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_sources(msg["content"], msg.get("sources", []))

if user_input := st.chat_input("Ask a question about the paper..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = answer_with_sources(user_input)
        st.markdown(answer)
        show_sources(answer, sources)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

with st.sidebar:
    st.markdown("### About this app")
    st.markdown("""
    This RAG chatbot answers questions over a 19-page research paper using:
    - **LangChain** for pipeline orchestration
    - **Sentence-Transformers** (all-MiniLM-L6-v2) for embeddings
    - **Pinecone** as the vector database
    - **gpt-oss-20b via Groq** as the LLM
    - **Streamlit** for the web interface

    Answers cite their source pages. Evaluated on a 30-question test set:
    22/30 correct, 0/3 hallucinated answers on out-of-scope questions.
    """)
    st.divider()
    st.markdown("Built by Sameeksha Mehrotra")
    st.markdown("[GitHub](https://github.com/sameeksha-mehrotra/rag-chatbot)")