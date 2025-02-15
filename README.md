# idp-search-rag
RAG Implementation for contextual search of India Data Portal Datasets

# Ideation and Design

The following are the primary design decisions for the RAG implementation in this repository:

1. Using a document store to store the dataset metadata
2. Using a vector store, chroma DB, to store the dataset embeddings for efficient similarity search
3. Utilizing the power of the LlamaIndex library to create a vector database and language model for efficient and contextual search
4. Using a language model to generate responses to user queries and provide contextually relevant information
5. Fast API for get requests and post requests of user queries

# To-Do
1. Scoring and ranking of results for retrieving only relevant results
2. A bug now, reload of Fast API app duplicates the documents in the chroma db, resulting in metadata length more than chunk size error. This is due to the fact that the app is reloaded and the documents are not removed from the chroma db. This is a temporary fix, and have to be fixed in future releases.
Workaround for now: Remove the chroma db before reloading the app, and run src/data_gatherer.py to create a new index

## Pre-Requisites
1. Groq LLM API Key, for llama3-70b-8192 model. Create an account on https://groq.ai/
2. Install poetry and all dependencies
   1. poetry config virtualenvs.in-project true
   2. poetry install
3. Create a .env file with the following variables:
   1. API_KEY=your_groq_api_key

## Usage:
1. Run `poetry run python src/data_gatherer.py` to create the index with vector embeddings stored in chroma db.
2. Run `poetry run uvicorn src.app:app --reload` to start the FastAPI server.

You should see the agent available on http://127.0.0.1:8000/search?query=agriculture

Responses will be in the format:
```
{
"id": "unknown",
"text": "company registrations ministry corporate affairs (mca) curates rich collection datasets illuminate corporate landscape india. within trove, one discern distribution indian companies across states, reflecting pulse nation's domestic corporate sector. equally revealing dataset foreign company registrations, showcasing global interest investment india's diverse regions. complementing data limited liability partnerships (llps), offers insights choice modern business entity across states. together, datasets provide holistic view india's dynamic business environment, aiding stakeholders navigating country's corporate maze. company-registrations ministry corporate affairs ['economy'] resource name: limited liability partnership(llps) company registrations new, format: csv, description: dataset contains information llp company registrations, including details month year registration, llp identification number (llpin), llp name, date registration, activity code, activity description., data_insights: general data insights drawn dataset :temporal trends: examining dataset different months years reveal trends llp registrations time.industry activity: analyzing industry activity descriptions give overview sectors llps prevalent,methodology: ministry corporate affairs gives monthly report registration companys, data_usage: ,frequency: monthly, sku: mca-llp_registrations_new-ol-mn-aaa,data_last_updated: 2025-01-31, data_retreival_date: 2025-02-05",
"metadata": {
"sku": "mca-llp_registrations_new-ol-mn-aaa"
},
"link": "https://dev.indiadataportal.com/p/company-registrations/r/mca-llp_registrations_new-ol-mn-aaa"
}
```
May be, learnings while doing this can be included.