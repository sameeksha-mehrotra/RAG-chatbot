from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import pinecone
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env file
load_dotenv()

# Step 1: Load your PDF
loader = PyPDFLoader("data/document.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# Step 2: Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# Step 3: Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings_list = embeddings.embed_documents([text.page_content for text in chunks])
print(f"Created embeddings for {len(embeddings_list)} chunks")

# Step 4: Store in Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

if "rag-chatbot" not in pc.list_indexes().names():
    pc.create_index(
        name="rag-chatbot",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Pinecone index created")

vectorstore = PineconeVectorStore.from_documents(chunks, embeddings,index_name="rag-chatbot")
print("Documents uploaded to Pinecone")