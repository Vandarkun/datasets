import json
import os
import pickle
import faiss  # 原生 faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LangChainFAISS

# === 配置 ===
DATA_PATH = '../data/tmdb_movie_metadata.jsonl'
OUTPUT_FOLDER = '../data/faiss'
INDEX_FILE = os.path.join(OUTPUT_FOLDER, 'movies.faiss')
META_FILE = os.path.join(OUTPUT_FOLDER, 'movies.pkl')
MODEL_PATH = '../model/all-MiniLM-L6-v2'  # 你的本地模型路径

def build_pure_vector_store():
    # 1. 加载数据
    print("正在加载数据...")
    texts = []
    metadata_list = []
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.strip()
                if not line: continue
                item = json.loads(line)
                
                # 提取字段
                title = item.get('title', 'Unknown')
                overview = item.get('overview', '')
                
                def list_to_str(key):
                    val = item.get(key, [])
                    return ", ".join(val) if isinstance(val, list) else str(val)

                # 构建文本内容 (Context)
                text = (
                    f"Title: {title}. "
                    f"Genres: {list_to_str('genres')}. "
                    f"Director: {list_to_str('director')}. "
                    f"Cast: {list_to_str('cast')}. "
                    f"Keywords: {list_to_str('keywords')}. "
                    f"Overview: {overview}"
                )
                
                # 构建元数据 (Metadata)
                meta = {
                    "title": title,
                    "genres": item.get('genres', []),
                    "director": item.get('director', []),
                    "cast": item.get('cast', []),
                    "overview": overview
                }
                
                texts.append(text)
                metadata_list.append({
                    "page_content": text,  # 把内容也存进 pickle，方便加载时还原
                    "metadata": meta
                })
                
            except json.JSONDecodeError:
                continue

    if not texts:
        print("错误：没有数据！")
        return

    # 2. 计算向量 (使用 LangChain 的工具方便计算)
    print(f"正在加载模型: {MODEL_PATH} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print(f"正在计算 {len(texts)} 条数据的向量...")
    # 我们先用 LangChain 临生成一个库，只为了利用它的计算逻辑
    temp_db = LangChainFAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=[m["metadata"] for m in metadata_list]
    )

    # 3. 【关键步骤】提取原生 FAISS 索引和纯数据保存
    # 这样就绕过了 LangChain 的 save_local，避免了 Pydantic 报错
    print(f"正在保存纯净版索引到: {OUTPUT_FOLDER}")
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # A. 保存物理索引 (FAISS 原生格式)
    faiss.write_index(temp_db.index, INDEX_FILE)
    
    # B. 保存数据列表 (纯 Python 字典，绝无 Pydantic 版本问题)
    with open(META_FILE, 'wb') as f:
        pickle.dump(metadata_list, f)

    print("构建完成！生成了兼容性最强的文件。")

if __name__ == "__main__":
    build_pure_vector_store()