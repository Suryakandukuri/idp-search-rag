from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import chromadb
from llama_index import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import os




load_dotenv()

llm = Groq(model="llama3-70b-8192", api_key=os.getenv('API_KEY'))

# storage_context = StorageContext.from_defaults(
#     docstore=SimpleDocumentStore.from_persist_dir(persist_dir="../llama_index.json"),
#     vector_store=SimpleVectorStore.from_persist_dir(
#         persist_dir="../llama_index.json",
#     ),
#     index_store=SimpleIndexStore.from_persist_dir(persist_dir="<persist_dir>"),
# )
# indices = load_indices_from_storage(storage_context)

# Load the ChromaDB collection from disk
chroma_client = chromadb.PersistentClient(path="chroma_db")
chroma_collection = chroma_client.get_collection("idp_search")


# Define embedding function
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Set up ChromaVectorStore
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(storage_context=storage_context, embed_model=embed_model)


# Define FastAPI app
app = FastAPI()

class SearchRequest(BaseModel):
    query: str

class SearchResult(BaseModel):
    id: str
    text: str
    metadata: dict
    link: str

@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    query_engine = index.as_query_engine()
    response = query_engine.query(request.query)
   
    # Format the response to include dataset links
    results = []
    for doc in response:
        link = f"www.dev.indiadataportal.com/p/{doc.metadata['package_name']}/r/{doc.metadata['sku']}"
        results.append({
            "id": doc.id,
            "text": doc.text,
            "metadata": doc.metadata,
            "link": link
        })
   
    return results

@app.post("/refresh")
def refresh_data():
    # Re-fetch and re-index data
    gather_embed_and_index()
    return {"message": "Data refreshed and index updated."}