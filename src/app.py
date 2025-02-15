from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
from src.data_gatherer import gather_embed_and_index
from llama_index.core.schema import Document
from llama_index.core.node_parser.text import SentenceSplitter
import os
import json

load_dotenv()

llm = Groq(model="llama3-70b-8192", api_key=os.getenv("API_KEY"))


# Load metadata store
try:
    with open("metadata_store.json", "r") as f:
        metadata_store = json.load(f)
except FileNotFoundError:
    metadata_store = {}

# Load ChromaDB collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    chroma_collection = chroma_client.get_collection("idp_search")
    existing_docs = chroma_collection.count()
    print(existing_docs)
except:
    existing_docs = 0

print(f"Existing documents in ChromaDB: {existing_docs}")

if existing_docs == 0:
    print("No existing vector database found. Go Rebuild index...")
else:
    print(f"Loading existing ChromaDB with {existing_docs} documents.")
    # Load Storage Context
    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection)
    )

    # Define embedding function
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    retrieved_docs = chroma_collection.get(include=["documents", "metadatas"])
    documents = [
        Document(text=doc, extra_info=meta)
        for doc, meta in zip(retrieved_docs["documents"], retrieved_docs["metadatas"])
    ]
    # print(documents[1].text)
    # print(documents[1].extra_info)
    # Increase chunk size
    sentence_splitter = SentenceSplitter(chunk_size=2048)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[sentence_splitter],
        include_metadata_in_chunk=False
    )


# Define FastAPI app
app = FastAPI()


class SearchRequest(BaseModel):
    query: str


class SearchResult(BaseModel):
    id: str
    text: str
    metadata: dict
    link: str


@app.get("/search", response_model=List[SearchResult])
@app.post("/search", response_model=List[SearchResult])
async def search(query: str = Query(..., description="Search query")):
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=20)
    response = query_engine.retrieve(query)
    print(f"Retrieved {len(response)} documents for query: {query}")

    results = []
    print(response)
    # Ensure response is iterable
    if isinstance(response, list):
        docs = response
    else:
        docs = [response]  # Wrap in a list if it's a single object

    for doc in docs:
        print(doc)
        sku = doc.metadata.get("sku", "")
        full_metadata = metadata_store.get(sku, {})
        link = f"https://dev.indiadataportal.com/p/{full_metadata.get('package_name', '')}/r/{full_metadata.get('sku', '')}"
        results.append(
            {
                "id": doc.id if hasattr(doc, "id") else "unknown",
                "text": doc.text,
                "metadata": doc.metadata,
                "link": link,
            }
        )
    return results
