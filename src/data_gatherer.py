# from ckanapi import RemoteCKAN
import requests
import nltk
from nltk.corpus import stopwords
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.storage import StorageContext
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Create Chroma client and collection

chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    chroma_collection = chroma_client.get_collection("idp_search")
except:
    chroma_collection = chroma_client.create_collection("idp_search")

llm = Groq(model="llama3-70b-8192", api_key=os.getenv("API_KEY"))


nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# creating a dictionary for metadata_store
metadata_store = {}

# CKAN API Data Retrieval
def fetch_ckan_package_data():
    api_url = "https://ckandev.indiadataportal.com/api/3/action/package_search?q=organization%3Aidp-organization&rows=1000"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    }
    # Make the request
    response = requests.get(api_url, headers=headers)

    if response.status_code != 200:
        print(f"Error: Received HTTP {response.status_code}")
        print(f"Response text: {response.text}")
    else:
        try:
            json_data = response.json()
        except requests.exceptions.JSONDecodeError:
            print("Error: Response is not valid JSON")
            print(f"Raw response: {response.text}")

    if response.status_code == 200 and json_data.get("success"):
        packages = json_data["result"]["results"]
        documents = []
    for package in packages:
        for resource in package["resources"]:
            # datastore_info_texts = []
            sku = resource.get("sku", "")
            resource_text = f"Resource Name: {resource.get('name', '')}, Format: {resource.get('format', '')}, Description: {resource.get('description', '')}, Data_Insights: {resource.get('data_insights', '')},methodology: {resource.get('methodology', '')}, Data_Usage: {resource.get('data_usage', '')},frequency: {resource.get('frequency', '')}, sku: {resource.get('sku', '')},data_last_updated: {resource.get('data_last_updated', '')}, data_retreival_date: {resource.get('data_retreival_date', '')}"
            # api_url = f"https://ckandev.indiadataportal.com/api/3/action/datastore_info?id={resource['id']}"
            # response = requests.get(api_url).json()
            # rows = response.get("result", {}).get("records", [])
            # datastore_info_texts.append(" ".join(str(row) for row in rows))
            # Store full metadata separately
            metadata = {
                "package_id": package["id"],
                "title": package["title"],
                "url": package["url"],
                "package_name": package["name"],
                "sku": sku
            }
            metadata_store[sku] = metadata
        combined_text = f"{package['title']} {package['notes']} {package['name']} {package['source_name']} {package['sector']} {resource_text}"
        
        documents.append(
            {
                "text": preprocess_text(combined_text),
                "metadata": {
                    "sku": sku
                },
            }
        )
            # Save metadata store to disk
    with open("metadata_store.json", "w") as f:
        json.dump(metadata_store, f)
    return documents


# Fetch and combine metadata from resources
def fetch_resource_details(resources):
    resource_texts = []
    sku_list = []
    for resource in resources:
        sku_list.append(resource.get("sku", ""))
        resource_text = f"Resource Name: {resource.get('name', '')}, Format: {resource.get('format', '')}, Description: {resource.get('description', '')}, Data_Insights: {resource.get('data_insights', '')},methodology: {resource.get('methodology', '')}, Data_Usage: {resource.get('data_usage', '')},frequency: {resource.get('frequency', '')}, sku: {resource.get('sku', '')},data_last_updated: {resource.get('data_last_updated', '')}, data_retreival_date: {resource.get('data_retreival_date', '')}"
        resource_texts.append(resource_text)
    return " ".join(resource_texts), sku_list


# Datastore SQL API Retrieval (Example for query fetching columns)
def fetch_datastore_info(resources):
    datastore_info_texts = []
    for resource in resources:
        api_url = f"https://ckandev.indiadataportal.com/api/3/action/datastore_info?id={resource['id']}"
        response = requests.get(api_url).json()
        rows = response.get("result", {}).get("records", [])
        datastore_info_texts.append(" ".join(str(row) for row in rows))
        return datastore_info_texts


# Custom preprocessing logic
def preprocess_text(text):
    # Convert to lowercase and strip extra spaces
    text = text.lower().strip()

    # Tokenize the text and remove stopwords
    tokens = [word for word in text.split() if word not in stop_words]

    # Join the tokens back into a string
    return " ".join(tokens)


# Convert the documents into LlamaIndex's document structure
def create_index(documents, chroma_collection=chroma_collection):

    # Define embedding function

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    indexed_documents = [
        Document(text=doc["text"], extra_info=doc["metadata"]) for doc in documents
    ]

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        indexed_documents, storage_context=storage_context, embed_model=embed_model
    )

    # Persist ChromaDB
    storage_context.persist(persist_dir="chroma_db")

    return index


def gather_embed_and_index():
    # Step 1: Fetch data
    documents = fetch_ckan_package_data()
    if not documents:
        print("No documents retrieved from CKAN. Skipping indexing.")
        return None
    chroma_client.delete_collection("idp_search")  # Deletes the entire collection
    chroma_collection = chroma_client.create_collection("idp_search")  # Recreate it

    # Step 3: Embeddings, Chroma DB, Create the index using LlamaIndex
    index = create_index(documents, chroma_collection=chroma_collection)
    # Persist both LlamaIndex and ChromaDB
    index.storage_context.persist(persist_dir="chroma_db")
    print(f"Index built with {len(documents)} documents and persisted to ChromaDB.")

    return index


if __name__ == "__main__":
    index = gather_embed_and_index()
    # index.storage_context.persist(persist_dir="chroma_db")
    print("Data gathered, index created, vectordb persisted and chromadb saved.")
