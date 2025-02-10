from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json

def generate_embeddings_with_metadata(processed_entries, metadata_entries):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(processed_entries, show_progress_bar=True)
    return np.array(embeddings), metadata_entries


def update_vector_store_with_metadata(embeddings, metadata_entries):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save the metadata separately
    with open('metadata.json', 'w') as f:
        json.dump(metadata_entries, f)

    faiss.write_index(index, "vector_store_with_metadata.index")