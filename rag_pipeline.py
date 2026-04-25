#pip install datasets sentence-transformers chromadb pandas

 
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd

# CONFIG 
 
HF_DATASET_NAME = "dralsarrani/prompt_safety_with_synthetic"  
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # fast, free, good enough
CHROMA_DIR       = "./chroma_db"         # local folder, created automatically
COLLECTION_NAME  = "safety_prompts"
TOP_K            = 5                     # how many similar prompts to retrieve


 
# 1 LOAD DATASET
 
def load_safety_dataset():
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset(HF_DATASET_NAME, cache_dir="./hf_cache", download_mode="force_redownload")
    df = dataset["train"].to_pandas()
 
    # Normalise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]
 
    # Keep only rows with valid prompt + label
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].isin([1.0, 0.0])]
    df = df.reset_index(drop=True)
 
    print(f"  Loaded {len(df)} rows  |  SAFE: {(df.label==0).sum()}  UNSAFE: {(df.label==1).sum()}")
    return df
 
 
# 2 BUILD CHROMA VECTOR STORE 
 
def build_vector_store(df: pd.DataFrame):
    print("Building vector store...")
    model  = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
 
    # Delete existing collection so we start fresh on rebuild
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
 
    collection = client.create_collection(COLLECTION_NAME)
 
    prompts = df["text"].tolist()
    labels  = df["label"].tolist()
    ids     = [str(i) for i in range(len(prompts))]
 
    # Embed in batches of 512 to avoid memory issues on large datasets
    batch_size = 512
    all_embeddings = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False).tolist()
        all_embeddings.extend(embeddings)
        print(f"  Embedded {min(i + batch_size, len(prompts))}/{len(prompts)}")

    batch_size_chroma = 5000
    for i in range(0, len(ids), batch_size_chroma):
        batch_ids = ids[i : i + batch_size_chroma]
        batch_embeds = all_embeddings[i : i + batch_size_chroma]
        batch_docs = prompts[i : i + batch_size_chroma]
        batch_metadatas = [{"label": l} for l in labels[i : i + batch_size_chroma]]
        collection.add(
        ids=batch_ids,
        embeddings=batch_embeds,
        documents=batch_docs,
        metadatas=batch_metadatas
        )
    
 
    print(f"  Stored {collection.count()} vectors in Chroma")
    return collection, model
 
 # 3 RETRIEVAL FUNCTION 
 
def retrieve_similar(query: str, collection, model, top_k: int = TOP_K):
    """
    Given a new prompt, return the top_k most similar prompts
    from the dataset with their labels and similarity scores.
    """
    query_embedding = model.encode([query]).tolist()
 
    results = collection.query(
        query_embeddings = query_embedding,
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"],
    )
 
    similar = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similar.append({
            "prompt":     doc,
            "label":      meta["label"],
            "similarity": round(1 - dist, 3),   # cosine distance → similarity
        })
 
    return similar
 

 # 4 LOAD EXISTING STORE (skip rebuild if already done) 
 
def load_vector_store():
    """Load an already-built Chroma store without re-embedding."""
    model      = SentenceTransformer(EMBEDDING_MODEL)
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Loaded existing vector store ({collection.count()} vectors)")
    return collection, model
 