from fastapi import FastAPI
import faiss
import numpy as np
from dotenv import load_dotenv
import os
import json
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from data_gatherer import gather_and_process_data

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Load the FAISS index and metadata
index = faiss.read_index("vector_store_with_metadata.index")
with open('metadata.json', 'r') as f:
    metadata_entries = json.load(f)

# Load Sentence Transformer for generating query embeddings
query_model = SentenceTransformer('all-MiniLM-L6-v2')

# Perform the search
def retrieve_with_metadata(query_embedding, top_k=5):
    try:
        distances, indices = index.search(np.array([query_embedding]), top_k)
        if not indices.any():
            return []
        retrieved_sources = [metadata_entries[i] for i in indices[0] if i < len(metadata_entries)]
        return retrieved_sources
    except Exception as e:
        return {"error": str(e)}


@app.get("/search")
def search(query: str):
    # Generate embedding for the query
    query_embedding = query_model.encode([query])

    # Retrieve relevant datasets
    retrieved_sources = retrieve_with_metadata(query_embedding)

    # Initialize OpenAI LLM and generate a response
    llm = OpenAI(api_key=openai_api_key)  # Ensure this is imported properly
    openai_response = llm(query)

    return {
        "query": query,
        "generated_response": openai_response,
        "datasets": retrieved_sources
    }

@app.post("/refresh")
def refresh_data():
    # Re-fetch and re-index data
    gather_and_process_data()
    return {"message": "Data refreshed and index updated."}