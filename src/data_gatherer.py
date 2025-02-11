# from ckanapi import RemoteCKAN
import requests
from embeddings_vectordb import *
import json
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
import torch
from llama_index.core import Settings, VectorStoreIndex, Document
import llama_index
from llama_index.core import Settings
from llama_index.llms.groq import Groq


llm = Groq(model="llama3-70b-8192", api_key="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")

# Settings.llm = Ollama(model="llama2", request_timeout=120.0)
# Settings.embed_model = HuggingFaceEmbedding(
#     model_name="BAAI/bge-small-en-v1.5"
# )

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# CKAN API Data Retrieval
def fetch_ckan_package_data():
    api_url = "https://ckandev.indiadataportal.com/api/3/action/package_search?q=organization%3Aidp-organization&rows=1000"
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
    # Make the request
    response = requests.get(api_url, headers=headers)

    if response.status_code != 200:
        print(f"Error: Received HTTP {response.status_code}")
        print(f"Response text: {response.text}")
    else:
        try:
            print("success")
            json_data = response.json()
        except requests.exceptions.JSONDecodeError:
            print("Error: Response is not valid JSON")
            print(f"Raw response: {response.text}")

    if response.status_code == 200 and json_data.get('success'):
        packages = json_data['result']['results']
        documents = []
    for package in packages:
        resource_details = fetch_resource_details(package['resources']) 
        datastore_info = fetch_datastore_info(package['resources'])
        combined_text = f"{package['title']} {package['notes']} {resource_details} {datastore_info}"
        documents.append({"text": preprocess_text(combined_text), "metadata": {"package_id": package["id"], "title": package["title"], "url": package["url"]}})
    return documents

# Fetch and combine metadata from resources
def fetch_resource_details(resources):
    resource_texts = []
    for resource in resources:
        resource_text = f"Resource Name: {resource.get('name', '')}, Format: {resource.get('format', '')}, Description: {resource.get('description', '')}"
        resource_texts.append(resource_text)
    return ' '.join(resource_texts)

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
    return ' '.join(tokens)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Generate embeddings using BERT
def generate_bert_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the mean of token embeddings as the document embedding
            last_hidden_state = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(last_hidden_state.squeeze().numpy())
    return embeddings


# Convert the documents into LlamaIndex's document structure
def create_index(documents):
    indexed_documents = [Document(text=doc["text"], extra_info=doc["metadata"]) for doc in documents]
    
    # Create the service context and index
    service_context = llama_index.settings.Settings.from_defaults()
    index = VectorStoreIndex.from_documents(indexed_documents, service_context=service_context)
    
    # Save the index
    index.storage_context.persist("llama_index.json")
    return index

def gather_embed_and_index():
    # Step 1: Fetch data
    documents = fetch_ckan_package_data()
    texts = [doc["text"] for doc in documents]

    # Step 2: Generate BERT embeddings (if needed for custom processing)
    embeddings = generate_bert_embeddings(texts)  # Optional if you wish to modify indexing

    # Step 3: Create the index using LlamaIndex
    create_index(documents)


if __name__ == "__main__":
    gather_embed_and_index()