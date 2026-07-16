"""
rag_retriever.py
Builds a FAISS vector store from historical FMCG campaigns using LangChain.
Run this script once to generate the RAG vector store.
"""

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration
CAMPAIGN_PATH = "data/historical_campaigns.txt"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OUTPUT_DIR = "models/rag_index"

def build_campaign_vectorstore():
    print("📄 Loading campaign data...")
    
    # Read the campaigns file
    with open(CAMPAIGN_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split into individual campaigns (using CAMPAIGN_ID as separator)
    # This ensures each chunk is a complete campaign
    chunks = text.split("CAMPAIGN_ID:")
    chunks = ["CAMPAIGN_ID:" + chunk for chunk in chunks if chunk.strip()]
    
    # If no splits found (fallback), use recursive splitter
    if len(chunks) <= 1:
        print("⚠️ Using recursive splitter as fallback...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " "]
        )
        chunks = text_splitter.split_text(text)
    
    print(f"✅ Loaded {len(chunks)} campaign chunks")
    
    # Initialize embedding model
    print("🧠 Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Build FAISS vector store
    print("🔍 Building RAG vector store...")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    # Save locally
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vectorstore.save_local(OUTPUT_DIR)
    
    print(f"✅ RAG vector store saved to {OUTPUT_DIR}")
    
    return vectorstore

def load_retriever():
    """
    Utility function to load the saved retriever.
    Used by the FastAPI server.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        OUTPUT_DIR, 
        embeddings, 
        allow_dangerous_deserialization=True  # Required for loading saved FAISS
    )
    return vectorstore.as_retriever(search_kwargs={"k": 2})

if __name__ == "__main__":
    build_campaign_vectorstore()
    print("\n🎉 RAG vector store built successfully!")
    print(f"   - Saved to: {OUTPUT_DIR}/")
    print("\n💡 Now you can run your FastAPI server!")