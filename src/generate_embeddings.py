"""
generate_embeddings.py
Builds FAISS index and KMeans clusters from FMCG customer data.
Run this script once to generate the saved artifacts.
"""

import pandas as pd
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuration
DATA_PATH = "data/fmcg_customers.csv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
N_CLUSTERS = 4  # We want 4 distinct FMCG segments
OUTPUT_DIR = "models"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_customer_index():
    print("📊 Loading customer data...")
    df = pd.read_csv(DATA_PATH)
    
    # Create rich textual profile for each customer
    df['profile'] = df.apply(
        lambda row: (
            f"Age: {row['Age']}, Income: {row['Income_LPA']}LPA, "
            f"Family: {row['Family_Size']}, City: {row['City_Tier']}, "
            f"Channel: {row['Primary_Channel']}, "
            f"Snacks_Spend: {row['Snacks_Spend']}, "
            f"Beverage_Spend: {row['Beverages_Spend']}, "
            f"Household_Spend: {row['Household_Spend']}, "
            f"PersonalCare_Spend: {row['PersonalCare_Spend']}, "
            f"Promo: {row['Promo_User']}, Organic: {row['Organic_Buyer']}"
        ),
        axis=1
    )
    
    print("🧠 Generating embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(df['profile'].tolist(), show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # --- Build FAISS Index (for similarity search) ---
    print("🔍 Building FAISS index...")
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    
    # Save FAISS index
    faiss_path = os.path.join(OUTPUT_DIR, "customer_index.faiss")
    faiss.write_index(faiss_index, faiss_path)
    print(f"✅ FAISS index saved to {faiss_path}")
    
    # --- KMeans Clustering (for segment labels) ---
    print("📌 Performing KMeans clustering...")
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_embeddings)
    
    # Save KMeans model and scaler
    kmeans_path = os.path.join(OUTPUT_DIR, "kmeans.pkl")
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    
    with open(kmeans_path, 'wb') as f:
        pickle.dump(kmeans, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Add cluster labels to dataframe
    df['cluster'] = cluster_labels
    
    # Map cluster IDs to business-friendly segment names
    segment_mapping = {
        0: "🏙️ Metro Premium Buyers (High Income, High Spend)",
        1: "🛒 Kirana Value Hunters (Budget-Conscious, Tier 2/3)",
        2: "🌿 Health & Organic Enthusiasts (Mid-High Income, Organic)",
        3: "📱 Digital Gen-Z Snackers (Online, Low Age, High Promo)"
    }
    df['segment_name'] = df['cluster'].map(segment_mapping)
    
    # Save metadata (cluster mappings, customer IDs, profiles)
    metadata = {
        'customer_ids': df['CustomerID'].tolist(),
        'profiles': df['profile'].tolist(),
        'clusters': df['cluster'].tolist(),
        'segment_names': df['segment_name'].tolist(),
        'cluster_to_name': segment_mapping
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, "customer_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Metadata saved to {metadata_path}")
    print("\n📊 Cluster Distribution:")
    print(df['segment_name'].value_counts())
    
    # Save the full dataframe for reference
    df.to_csv(os.path.join(OUTPUT_DIR, "customer_with_clusters.csv"), index=False)
    
    return df

if __name__ == "__main__":
    build_customer_index()
    print("\n🎉 All artifacts generated successfully!")
    print("   - models/customer_index.faiss")
    print("   - models/kmeans.pkl")
    print("   - models/scaler.pkl")
    print("   - models/customer_metadata.pkl")
    print("   - models/customer_with_clusters.csv")