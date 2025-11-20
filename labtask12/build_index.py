import glob
import re
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

COLUMNS = [
    'Chapter_Number', 'Chapter_English', 'Chapter_Arabic',
    'Section_Number', 'Section_English', 'Section_Arabic',
    'Hadith_Number',
    'English_Hadith', 'English_Isnad', 'English_Matn', 'English_Grade',
    'Arabic_Hadith', 'Arabic_Isnad', 'Arabic_Matn', 'Arabic_Grade'
]


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean(csv_path_pattern: str):
    files = sorted(glob.glob(csv_path_pattern, recursive=True))
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f, names=COLUMNS, skiprows=1, encoding='utf-8')
        except Exception:
            df = pd.read_csv(f, names=COLUMNS, skiprows=1, encoding='latin-1')
        df['English_Hadith'] = df['English_Hadith'].fillna('')
        df['Cleaned_Hadith'] = df['English_Hadith'].apply(clean_text)
        rows.append(df)
    if not rows:
        raise FileNotFoundError('No CSV files found for the pattern: ' + csv_path_pattern)
    combined = pd.concat(rows, ignore_index=True)
    combined = combined[combined['Cleaned_Hadith'].str.strip().astype(bool)].reset_index(drop=True)
    return combined


def build_embeddings(df: pd.DataFrame, model_name='sentence-transformers/all-MiniLM-L6-v2', batch_size=256):
    model = SentenceTransformer(model_name)
    texts = df['Cleaned_Hadith'].tolist()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    return embeddings


def build_faiss_index(embeddings: np.ndarray, index_path: str):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  
    faiss.write_index(index, index_path)
    return index


def main(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(data_dir, '**', '*.csv')
    print('Loading CSVs from pattern:', pattern)
    df = load_and_clean(pattern)
    print('Total hadiths after cleaning:', len(df))
    print('Building embeddings...')
    embeddings = build_embeddings(df)
    emb_path = os.path.join(output_dir, 'hadith_embeddings.npy')
    np.save(emb_path, embeddings)
    print('Saved embeddings to', emb_path)
    index_path = os.path.join(output_dir, 'faiss_index.index')
    build_faiss_index(embeddings, index_path)
    print('Saved faiss index to', index_path)
    meta_path = os.path.join(output_dir, 'hadith_metadata.parquet')
    df.reset_index(drop=True).to_parquet(meta_path, index=False)
    print('Saved metadata to', meta_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='LK-Hadith-Corpus', help='root folder containing CSVs')
    parser.add_argument('--output_dir', type=str, default='artifacts', help='where to save index/embeddings/metadata')
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
