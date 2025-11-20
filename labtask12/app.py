# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import re
import os


app = Flask(__name__)
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
ARTIFACT_DIR = os.environ.get('ARTIFACT_DIR', 'artifacts')
TOP_K = 6


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


print('Loading model...')
model = SentenceTransformer(MODEL_NAME)
print('Loading metadata...')
meta_path = f"{ARTIFACT_DIR}/hadith_metadata.parquet"
if not os.path.exists(meta_path):
    raise FileNotFoundError(f"Metadata file not found: {meta_path}")
meta = pd.read_parquet(meta_path)
print('Loading embeddings...')
emb_path = f"{ARTIFACT_DIR}/hadith_embeddings.npy"
if not os.path.exists(emb_path):
    raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
embeddings = np.load(emb_path)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
print('Loading FAISS index...')
index_path = f"{ARTIFACT_DIR}/faiss_index.index"
if not os.path.exists(index_path):
    raise FileNotFoundError(f"FAISS index file not found: {index_path}")
index = faiss.read_index(index_path)

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query_api():
    data = request.json
    q = data.get('query', '')
    top_k = data.get('top_k', TOP_K)
    if not q:
        return jsonify({'error': 'Empty query'}), 400
    q_clean = clean_text(q)
    q_emb = model.encode([q_clean], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    D, I = index.search(q_emb, int(top_k))
    results = []
    for dist, idx in zip(D[0], I[0]):
        row = meta.iloc[int(idx)].to_dict()
        score = float(dist)
        results.append({
            'score': score,
            'Chapter_Number': row.get('Chapter_Number'),
            'Chapter_English': row.get('Chapter_English'),
            'Hadith_Number': row.get('Hadith_Number'),
            'English_Hadith': row.get('English_Hadith'),
            'English_Grade': row.get('English_Grade')
        })
    return jsonify({'query': q, 'results': results})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
