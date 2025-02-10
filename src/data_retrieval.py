import json
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
import faiss
import numpy as np
from langchain.llms import OpenAI

# Load metadata
with open('metadata.json', 'r') as f:
    metadata_entries = json.load(f)

# Load FAISS index
index = faiss.read_index("vector_store_with_metadata.index")

# Perform the search
def retrieve_with_metadata(query_embedding, top_k=5):
    distances, indices = index.search(np.array([query_embedding]), top_k)
    retrieved_sources = [metadata_entries[i] for i in indices[0]]
    return retrieved_sources

retriever = FAISS.load_local("/path/to/vector_store_with_metadata.index")
query_embedding = retriever.embedding_fn("Show me all datasets related to gender or women.")
retrieved_sources = retrieve_with_metadata(query_embedding)

# Generate final response
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(model="text-davinci-003"), chain_type="stuff", retriever=retriever)
response = qa_chain.run("Show me all datasets related to gender or women.")

# Print response with sources
print("Generated Response:", response)
print("Relevant Datasets:")
for source in retrieved_sources:
    print(f"Dataset Title: {source['title']}, URL: {source['url']}")