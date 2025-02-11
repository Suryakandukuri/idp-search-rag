from fastapi import FastAPI
from dotenv import load_dotenv
import os
from src.data_gatherer import gather_embed_and_index
from llama_index.llms.groq import Groq
from llama_index.core import load_indices_from_storage
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.storage import StorageContext





load_dotenv()

llm = Groq(model="llama3-70b-8192", api_key=os.getenv('API_KEY'))

storage_context = StorageContext.from_defaults(
    docstore=SimpleDocumentStore.from_persist_dir(persist_dir="../llama_index.json"),
    vector_store=SimpleVectorStore.from_persist_dir(
        persist_dir="../llama_index.json",
    ),
    index_store=SimpleIndexStore.from_persist_dir(persist_dir="<persist_dir>"),
)
indices = load_indices_from_storage(storage_context)





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

app = FastAPI()
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