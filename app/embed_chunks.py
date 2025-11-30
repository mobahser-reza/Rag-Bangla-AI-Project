import re
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Paths
chunk_path = 'data/chunks_2line.txt'
vector_store_path = 'app/vector_store_2line.pkl'

with open(chunk_path, 'r', encoding='utf-8') as f:
    text = f.read()

chunks = re.findall(r'---chunk_\d+---\n(.*?)(?=(---chunk_\d+---|$))', text, re.DOTALL)
chunk_texts = [c[0].strip() for c in chunks]

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

embeddings = model.encode(chunk_texts, show_progress_bar=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

vector_store = {
    'chunks': chunk_texts,
    'embeddings': embeddings,
    'faiss_index': index
}
with open(vector_store_path, 'wb') as f:
    pickle.dump(vector_store, f)

print(f"Total {len(chunk_texts)} chunks embedded and vector store saved to {vector_store_path}")