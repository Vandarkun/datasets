import json
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_PATH = '../data/tmdb_movie_metadata.jsonl'
OUTPUT_FOLDER = '../data/faiss'
INDEX_FILE = os.path.join(OUTPUT_FOLDER, 'movies.faiss')
META_FILE = os.path.join(OUTPUT_FOLDER, 'movies.pkl')
MODEL_PATH = '../model/all-MiniLM-L6-v2' 

def build_pure_faiss_index():
    print(f"正在读取数据: {DATA_PATH} ...")
    if not os.path.exists(DATA_PATH):
        print(f"错误: 找不到文件 {DATA_PATH}")
        return

    texts = []
    metadata_list = []
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.strip()
                if not line: continue
                item = json.loads(line)
                
                title = item.get('title', 'Unknown')
                overview = item.get('overview', '')
                
                def list_to_str(key):
                    val = item.get(key, [])
                    return ", ".join(val) if isinstance(val, list) else str(val)

                # 构建用于 Embedding 的文本
                text = (
                    f"Title: {title}. "
                    f"Genres: {list_to_str('genres')}. "
                    f"Director: {list_to_str('director')}. "
                    f"Cast: {list_to_str('cast')}. "
                    f"Keywords: {list_to_str('keywords')}. "
                    f"Overview: {overview}"
                )
                
                # 构建元数据字典
                meta_payload = {
                    "title": title,
                    "genres": item.get('genres', []),
                    "director": item.get('director', []),
                    "cast": item.get('cast', []),
                    "overview": overview,
                    "tmdb_id": item.get('tmdb_id')
                }
                
                texts.append(text)
                
                metadata_list.append({
                    "page_content": text,
                    "metadata": meta_payload
                })
                
            except json.JSONDecodeError:
                continue

    if not texts:
        print("错误: 数据集为空！")
        return
    print(f"共加载 {len(texts)} 条电影数据。")

    print(f"正在加载模型: {MODEL_PATH} ...")
    model = SentenceTransformer(MODEL_PATH, device='cpu') 

    print("正在计算向量 (这可能需要几分钟)...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    print("正在构建 FAISS 索引...")
    d = embeddings.shape[1]  # 向量维度 384
    
    index = faiss.IndexFlatIP(d) 
    index.add(embeddings)
    
    print(f"索引包含向量数: {index.ntotal}")

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print(f"保存索引到: {INDEX_FILE}")
    faiss.write_index(index, INDEX_FILE)

    print(f"保存元数据到: {META_FILE}")
    with open(META_FILE, 'wb') as f:
        pickle.dump(metadata_list, f)

    print("\n构建完成！")

if __name__ == "__main__":
    build_pure_faiss_index()