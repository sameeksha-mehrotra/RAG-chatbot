from dotenv import load_dotenv
import os
import requests
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
            "model": "meta-llama/llama-3.1-8b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.3
        }
        url = "https://router.huggingface.co/novita/v3/openai/chat/completions"
        response = requests.post(url, headers=headers, json=payload)
        print("Status code:", response.status_code)
        if response.status_code != 200:
            return f"API Error: {response.text}"
        result = response.json()
        return result["choices"][0]["message"]["content"]

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
    search_kwargs={"k": 3}
)

# LLM
llm = HFInferenceLLM(
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
    model_id="microsoft/Phi-3.5-mini-instruct"
)

# Prompt
prompt = PromptTemplate.from_template("""
Use the following context to answer the question.
If you don't know the answer, say so clearly.

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