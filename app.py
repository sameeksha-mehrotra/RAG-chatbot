from dotenv import load_dotenv
import os
import requests
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.llms import LLM
from typing import Optional, List

load_dotenv()

class HFInferenceLLM(LLM):
    api_key: str
    model_id: str = "meta-llama/llama-3.1-8b-instruct"

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.3
        }
        url = "https://router.huggingface.co/novita/v3/openai/chat/completions"
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            return f"API Error: {response.text}"
        return response.json()["choices"][0]["message"]["content"]

@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = PineconeVectorStore(
        index_name="rag-chatbot",
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    llm = HFInferenceLLM(api_key=os.getenv("HUGGINGFACE_API_KEY"))
    prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer from the context, say so clearly.

Context: {context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# UI
st.title("RAG Document Chatbot")
st.caption("Ask questions about your document. For example: 'What are the main points?' or 'Summarize the document.'")

chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chain.invoke(user_input)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})