from dotenv import load_dotenv
import os
import requests
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.llms import LLM
from typing import Optional, List

load_dotenv()

# Custom LLM that calls HF Inference API directly
class HFInferenceLLM(LLM):
    api_key: str
    model_id: str = "microsoft/Phi-3.5-mini-instruct"

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.3,
            "reasoning_effort": "low"
        }
        url = "https://api.groq.com/openai/v1/chat/completions"
        for attempt in range(6):
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 429:  # rate limited: wait, then retry
                time.sleep(10)
                continue
            break
        if response.status_code != 200:
            return f"API Error: {response.text}"
        return response.json()["choices"][0]["message"]["content"]

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to Pinecone
vectorstore = PineconeVectorStore(
    index_name="rag-chatbot",
    embedding=embeddings
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

# LLM
llm = HFInferenceLLM(
    api_key=os.getenv("GROQ_API_KEY"),
    model_id="microsoft/Phi-3.5-mini-instruct"
)

# Prompt
prompt = PromptTemplate.from_template("""
Answer the question using ONLY the context below. 
The answer is often a specific number, name, or list, so read every passage 
carefully before saying it is missing. Do not add facts, examples, or models that are 
not in the context. If the context truly doesn't contain the answer, say "I don't know."

Context: {context}

Question: {question}

Answer:""")

# Format docs helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    question = "What is the main topic of this document?"
    print("Question:", question)
    print("Answer:", chain.invoke(question))