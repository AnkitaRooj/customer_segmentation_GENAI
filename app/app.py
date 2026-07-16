from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from typing import List

app = FastAPI(title="FMCG GenAI Segmentation Engine")

# --- 1. Load Data & Build FAISS Customer Index ---
df = pd.read_csv("data/fmcg_customers.csv")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create textual profile for each customer
df['profile'] = df.apply(lambda row: f"Age: {row['Age']}, Income: {row['Income_LPA']}LPA, Family: {row['Family_Size']}, City: {row['City_Tier']}, Channel: {row['Primary_Channel']}, Snacks_Spend: {row['Snacks_Spend']}, Beverage_Spend: {row['Beverages_Spend']}, Organic: {row['Organic_Buyer']}", axis=1)

embeddings = model.encode(df['profile'].tolist())
dimension = embeddings.shape[1]
customer_index = faiss.IndexFlatL2(dimension)
customer_index.add(np.array(embeddings).astype('float32'))

# --- 2. Build RAG Vector Store from Historical Campaigns ---
with open("data/historical_campaigns.txt", "r", encoding="utf-8") as f:
    campaign_text = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(campaign_text)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
rag_vectorstore = LangChainFAISS.from_texts(chunks, embedding_model)
retriever = rag_vectorstore.as_retriever(search_kwargs={"k": 2})

# --- 3. Define LangChain Prompt for Positioning ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

prompt_template = PromptTemplate(
    input_variables=["segment_profile", "past_campaigns"],
    template="""
You are a senior FMCG marketing strategist.

Target Segment Profile: {segment_profile}

Past Successful Campaigns for similar segments (RAG context): {past_campaigns}

Generate a comprehensive marketing positioning strategy using the AIDA framework (Attention, Interest, Desire, Action).
Also suggest 3 hyper-localized ad headlines (mix of English & Hinglish).

Output:
Positioning Strategy:
Headlines:
"""
)
chain = LLMChain(llm=llm, prompt=prompt_template)

# --- 4. API Endpoint ---
class CustomerInput(BaseModel):
    age: int
    income_lpa: float
    family_size: int
    city_tier: str
    primary_channel: str
    snacks_spend: int
    beverages_spend: int
    household_spend: int
    personalcare_spend: int
    promo_user: str
    organic_buyer: str

@app.post("/segment-and-position")
async def segment_and_position(customer: CustomerInput):
    # 1. Create profile text
    profile = f"Age: {customer.age}, Income: {customer.income_lpa}LPA, Family: {customer.family_size}, City: {customer.city_tier}, Channel: {customer.primary_channel}, Snacks_Spend: {customer.snacks_spend}, Organic: {customer.organic_buyer}"
    
    # 2. Embed and search FAISS for nearest segment (Simulating Clustering)
    query_emb = model.encode([profile])
    distances, indices = customer_index.search(np.array(query_emb).astype('float32'), k=1)
    similar_customer = df.iloc[indices[0][0]]
    
    # 3. Build Segment Description
    segment_desc = f"Cluster profile: {similar_customer['profile']}. 
    This segment typically shops via {similar_customer['Primary_Channel']} in {similar_customer['City_Tier']} cities."
    
    # 4. RAG Retrieval: Fetch past campaigns based on segment similarity
    retrieved_docs = retriever.invoke(segment_desc)
    past_campaigns_text = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # 5. LangChain Generation
    final_output = chain.run({"segment_profile": segment_desc, "past_campaigns": past_campaigns_text})
    
    return {
        "assigned_segment": segment_desc,
        "retrieved_campaign_context": past_campaigns_text[:300] + "...",
        "positioning_and_copy": final_output,
        "nearest_customer_id": similar_customer['CustomerID']
    }

# Run: uvicorn api.app:app --reload