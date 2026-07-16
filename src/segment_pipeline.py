"""
segment_pipeline.py
Core pipeline that loads saved models and provides segmentation + RAG logic.
Used by api/app.py
"""

import os
import pickle
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.rag_retriever import load_retriever

# Configuration
MODEL_DIR = "models"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

class FMCSegmentationPipeline:
    """Main pipeline class for FMCG customer segmentation and positioning."""
    
    def __init__(self):
        print("🚀 Initializing FMCG Segmentation Pipeline...")
        
        # 1. Load embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # 2. Load FAISS index
        faiss_path = os.path.join(MODEL_DIR, "customer_index.faiss")
        self.faiss_index = faiss.read_index(faiss_path)
        
        # 3. Load KMeans and Scaler
        kmeans_path = os.path.join(MODEL_DIR, "kmeans.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        
        with open(kmeans_path, 'rb') as f:
            self.kmeans = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # 4. Load metadata
        metadata_path = os.path.join(MODEL_DIR, "customer_metadata.pkl")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        # 5. Load RAG retriever (campaigns)
        self.retriever = load_retriever()
        
        # 6. Segment name mapping
        self.segment_mapping = self.metadata['cluster_to_name']
        
        print("✅ Pipeline initialized successfully!")
        print(f"   - {len(self.metadata['customer_ids'])} customers indexed")
        print(f"   - {len(self.segment_mapping)} segments available")
    
    def _create_profile_text(self, customer_data):
        """Convert raw customer input into a profile text for embedding."""
        return (
            f"Age: {customer_data['age']}, Income: {customer_data['income_lpa']}LPA, "
            f"Family: {customer_data['family_size']}, City: {customer_data['city_tier']}, "
            f"Channel: {customer_data['primary_channel']}, "
            f"Snacks_Spend: {customer_data['snacks_spend']}, "
            f"Beverage_Spend: {customer_data['beverages_spend']}, "
            f"Household_Spend: {customer_data['household_spend']}, "
            f"PersonalCare_Spend: {customer_data['personalcare_spend']}, "
            f"Promo: {customer_data['promo_user']}, Organic: {customer_data['organic_buyer']}"
        )
    
    def assign_segment(self, customer_input):
        """
        Assign a segment to a new customer profile.
        
        Args:
            customer_input: dict with all customer features
            
        Returns:
            dict: Contains cluster_id, segment_name, and similar customer info
        """
        # 1. Create profile and embed
        profile_text = self._create_profile_text(customer_input)
        embedding = self.embedding_model.encode([profile_text])
        embedding = np.array(embedding).astype('float32')
        
        # 2. FAISS similarity search (find nearest existing customer)
        distances, indices = self.faiss_index.search(embedding, k=1)
        nearest_idx = indices[0][0]
        
        # 3. Predict cluster using KMeans (on the scaled embedding)
        scaled_emb = self.scaler.transform(embedding)
        cluster_id = self.kmeans.predict(scaled_emb)[0]
        
        # 4. Get segment name
        segment_name = self.segment_mapping.get(cluster_id, f"Segment_{cluster_id}")
        
        # 5. Get nearest customer details
        nearest_customer_id = self.metadata['customer_ids'][nearest_idx]
        nearest_profile = self.metadata['profiles'][nearest_idx]
        nearest_cluster = self.metadata['clusters'][nearest_idx]
        
        return {
            "cluster_id": int(cluster_id),
            "segment_name": segment_name,
            "nearest_customer_id": nearest_customer_id,
            "nearest_customer_profile": nearest_profile,
            "nearest_customer_cluster": int(nearest_cluster),
            "similarity_distance": float(distances[0][0])
        }
    
    def get_rag_context(self, segment_name, top_k=2):
        """
        Retrieve relevant historical campaigns using RAG.
        
        Args:
            segment_name: The segment label to query
            
        Returns:
            list: Retrieved campaign documents
        """
        # Use the segment name as the query to the RAG vector store
        docs = self.retriever.invoke(segment_name)
        return docs[:top_k]
    
    def get_full_positioning(self, customer_input, use_rag=True):
        """
        End-to-end pipeline: Get segment + RAG context + (Optional: return raw context).
        This is called by the FastAPI endpoint before passing to LangChain.
        
        Returns:
            dict: Full pipeline output
        """
        # Step 1: Segment assignment
        segment_info = self.assign_segment(customer_input)
        
        # Step 2: RAG retrieval (if enabled)
        rag_context = ""
        if use_rag:
            docs = self.get_rag_context(segment_info['segment_name'])
            rag_context = "\n\n".join([doc.page_content for doc in docs])
        
        return {
            "assigned_segment": segment_info,
            "rag_campaigns": rag_context,
            "nearest_profile": segment_info['nearest_customer_profile']
        }

# Global singleton instance for the FastAPI app to import
_pipeline_instance = None

def get_pipeline():
    """Singleton pattern to ensure we only load models once."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = FMCSegmentationPipeline()
    return _pipeline_instance

# Simple test if run directly
if __name__ == "__main__":
    print("🧪 Testing pipeline with sample input...")
    
    pipeline = get_pipeline()
    
    # Test customer (Customer C003 from our dataset)
    test_customer = {
        "age": 32,
        "income_lpa": 8.2,
        "family_size": 3,
        "city_tier": "Tier1",
        "primary_channel": "Online",
        "snacks_spend": 2100,
        "beverages_spend": 1500,
        "household_spend": 3400,
        "personalcare_spend": 2500,
        "promo_user": "Yes",
        "organic_buyer": "No"
    }
    
    result = pipeline.get_full_positioning(test_customer)
    
    print("\n📊 Segmentation Result:")
    print(f"   Segment: {result['assigned_segment']['segment_name']}")
    print(f"   Cluster ID: {result['assigned_segment']['cluster_id']}")
    print(f"   Nearest Customer: {result['assigned_segment']['nearest_customer_id']}")
    print("\n📝 RAG Context (Campaigns):")
    print(result['rag_campaigns'][:300] + "...")
    
    print("\n✅ Pipeline test completed successfully!")