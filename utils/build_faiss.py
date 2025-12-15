import json
import os
from typing import List

# LangChain 组件
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 配置路径 (根据你提供的路径调整)
DATA_PATH = '../data/tmdb_movie_metadata.jsonl'
OUTPUT_FOLDER = '../data/faiss'  # 注意：LangChain保存的是文件夹路径
INDEX_NAME = 'movies'            # 索引主文件名
MODEL_PATH = '../model/all-MiniLM-L6-v2' # 本地模型路径

def load_and_process_data(file_path) -> List[Document]:
    """
    读取 JSONL 并转换为 LangChain Document 对象列表
    """
    documents = []
    print(f"正在加载数据: {file_path} ...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                line = line.strip()
                if not line: continue
                
                item = json.loads(line)
                
                # 1. 提取字段
                title = item.get('title', 'Unknown Title')
                overview = item.get('overview', '')
                
                # 辅助函数：处理列表转字符串
                def list_to_str(key):
                    val = item.get(key, [])
                    return ", ".join(val) if isinstance(val, list) else str(val)

                genres_str = list_to_str('genres')
                director_str = list_to_str('director')
                cast_str = list_to_str('cast')
                keywords_str = list_to_str('keywords')

                # 2. 构建用于 Embedding 的文本内容 (Page Content)
                # 这是 LLM 真正“看”到并在向量空间里搜索的内容
                text_to_embed = (
                    f"Title: {title}. "
                    f"Genres: {genres_str}. "
                    f"Director: {director_str}. "
                    f"Cast: {cast_str}. "
                    f"Keywords: {keywords_str}. "
                    f"Overview: {overview}"
                )
                
                # 3. 构建元数据 (Metadata)
                # 这些数据不参与计算向量，但在检索结果中可以直接读取
                metadata = {
                    "title": title,
                    "genres": item.get('genres', []),
                    "director": item.get('director', []),
                    "cast": item.get('cast', []),
                    "overview": overview,
                }
                
                # 4. 创建 Document 对象
                doc = Document(page_content=text_to_embed, metadata=metadata)
                documents.append(doc)
                
            except json.JSONDecodeError:
                print(f"警告: 第 {line_num+1} 行 JSON 格式错误，已跳过。")
                continue

    print(f"共加载 {len(documents)} 条有效数据。")
    return documents

def build_vector_store():
    # 1. 加载数据
    docs = load_and_process_data(DATA_PATH)
    if not docs:
        print("错误: 未加载到任何数据。")
        return

    # 2. 加载本地 Embedding 模型 (使用 LangChain 的封装类)
    print(f"正在加载本地模型: {MODEL_PATH} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_PATH,
        model_kwargs={'device': 'cpu'}, # 如果有显卡改成 'cuda'
        encode_kwargs={'normalize_embeddings': True}
    )

    # 3. 构建 FAISS 索引 (这一步会自动计算向量并建立 Docstore)
    print("正在计算向量并构建索引 (这可能需要几分钟)...")
    vector_db = FAISS.from_documents(
        documents=docs,
        embedding=embeddings
    )

    # 4. 保存到本地 (LangChain 标准格式)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    print(f"正在保存到: {OUTPUT_FOLDER} (Index Name: {INDEX_NAME})...")
    vector_db.save_local(folder_path=OUTPUT_FOLDER, index_name=INDEX_NAME)

    print("\n构建成功！")
    print(f"生成文件: {os.path.join(OUTPUT_FOLDER, INDEX_NAME + '.faiss')}")
    print(f"生成文件: {os.path.join(OUTPUT_FOLDER, INDEX_NAME + '.pkl')}")

if __name__ == "__main__":
    build_vector_store()