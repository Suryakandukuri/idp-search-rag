from fastapi import FastAPI
import faiss
import numpy as np
from dotenv import load_dotenv
import os
import json
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from data_gatherer import gather_embed_and_index

from llama_index.llms.groq import Groq
from llama_index import GPTSimpleVectorIndex




load_dotenv()

llm = Groq(model="llama3-70b-8192", api_key=os.getenv('API_KEY'))

app = FastAPI()

index = GPTSimpleVectorIndex.load_from_disk('llama_index.json')

# Perform the search
# def retrieve_with_metadata(index, top_k=5):
#     try:
#         indices = index.search(np.array([query_embedding]), top_k)
#         if not indices.any():
#             return []
#         retrieved_sources = [metadata_entries[i] for i in indices[0] if i < len(metadata_entries)]
#         return retrieved_sources
#     except Exception as e:
#         return {"error": str(e)}


@app.get("/search")
def search(query: str):
    # Generate embedding for the query
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(query)

    return {
        "query": query,
        "generated_response": response
    }

@app.post("/refresh")
def refresh_data():
    # Re-fetch and re-index data
    gather_embed_and_index()
    return {"message": "Data refreshed and index updated."}