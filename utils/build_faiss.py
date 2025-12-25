import json
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class MovieFaissBuilder:
    """
    构建电影检索的 FAISS 索引。
    """

    def __init__(
        self,
        data_path: str = "/data/wdk/datasets/data/tmdb_movie_metadata.jsonl",
        output_folder: str = "/data/wdk/datasets/data/faiss",
        model_path: str = '/data/wdk/datasets/model/all-MiniLM-L6-v2',
        device: str = "cpu",
    ):
        self.data_path = data_path
        self.output_folder = output_folder
        self.index_file = os.path.join(output_folder, "movies.faiss")
        self.meta_file = os.path.join(output_folder, "movies.pkl")
        self.model_path = model_path
        self.device = device

    def build(self):
        print(f"正在读取电影数据: {self.data_path} ...")
        if not os.path.exists(self.data_path):
            print(f"错误: 找不到文件 {self.data_path}")
            return

        texts = []
        metadata_list = []

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    line = line.strip()
                    if not line: 
                        continue
                    item = json.loads(line)

                    title = item.get('title', 'Unknown')
                    overview = item.get('overview', '')

                    def list_to_str(key):
                        val = item.get(key, [])
                        return ", ".join(val) if isinstance(val, list) else str(val)

                    text = (
                        f"Title: {title}. "
                        f"Genres: {list_to_str('genres')}. "
                        f"Director: {list_to_str('director')}. "
                        f"Cast: {list_to_str('cast')}. "
                        f"Keywords: {list_to_str('keywords')}. "
                        f"Overview: {overview}"
                    )

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

        print(f"正在加载模型: {self.model_path} ...")
        model = SentenceTransformer(self.model_path, device=self.device)

        print("正在计算向量 (这可能需要几分钟)...")
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

        print("正在构建 FAISS 索引...")
        d = embeddings.shape[1]  # 维度

        index = faiss.IndexFlatIP(d)
        index.add(embeddings)

        print(f"索引包含向量数: {index.ntotal}")

        os.makedirs(self.output_folder, exist_ok=True)

        print(f"保存索引到: {self.index_file}")
        faiss.write_index(index, self.index_file)

        print(f"保存元数据到: {self.meta_file}")
        with open(self.meta_file, 'wb') as f:
            pickle.dump(metadata_list, f)

        print("\n电影索引构建完成！")


class MemoryFaissBuilder:
    """
    构建用户记忆的 FAISS 索引。
    支持输入为 JSON list 或 JSONL，每条包含 user_id + key_memories。
    """

    def __init__(
        self,
        profile_path: str = '/data/wdk/datasets/output/sample_profile.json',
        output_folder: str = "/data/wdk/datasets/data/faiss",
        model_path: str = '/data/wdk/datasets/model/all-MiniLM-L6-v2',
        device: str = "cpu",
        index_filename: str = "user_memories.faiss",
        meta_filename: str = "user_memories.pkl",
    ):
        self.profile_path = profile_path
        self.output_folder = output_folder
        self.model_path = model_path
        self.device = device
        self.index_file = os.path.join(output_folder, index_filename)
        self.meta_file = os.path.join(output_folder, meta_filename)

    def _iter_profiles(self):
        if not os.path.exists(self.profile_path):
            raise FileNotFoundError(f"Profile file not found: {self.profile_path}")

        with open(self.profile_path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("Profile JSON should be a list of profiles.")
                for obj in data:
                    yield obj
            elif first_char == "{":
                obj = json.load(f)
                if isinstance(obj, list):
                    for item in obj:
                        yield item
                else:
                    yield obj
            else:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line.strip())
                        yield obj
                    except json.JSONDecodeError:
                        continue

    def build(self):
        texts = []
        metadata_list = []

        print(f"读取用户画像: {self.profile_path}")
        for profile in self._iter_profiles():
            uid = profile.get("user_id")
            memories = profile.get("key_memories", []) or []
            if not uid or not memories:
                continue

            for mem in memories:
                title = mem.get("movie_title", "Unknown")
                rating = mem.get("rating", 0)
                mem_text = mem.get("memory_text", "")

                text = f"User: {uid}. Movie: {title}. Rating: {rating}. Memory: {mem_text}"
                texts.append(text)
                metadata_list.append({
                    "user_id": uid,
                    "movie_title": title,
                    "rating": rating,
                    "memory_text": mem_text
                })

        if not texts:
            print("错误: 未找到可用的记忆数据。")
            return
        print(f"共收集 {len(texts)} 条记忆，来自 {len(set([m['user_id'] for m in metadata_list]))} 个用户。")

        print(f"加载模型: {self.model_path}")
        model = SentenceTransformer(self.model_path, device=self.device)

        print("编码记忆向量...")
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

        print("构建记忆 FAISS 索引...")
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)

        os.makedirs(self.output_folder, exist_ok=True)
        print(f"保存索引: {self.index_file}")
        faiss.write_index(index, self.index_file)

        print(f"保存元数据: {self.meta_file}")
        with open(self.meta_file, "wb") as f:
            pickle.dump(metadata_list, f)

        print("记忆索引构建完成！")


if __name__ == "__main__":
    MemoryFaissBuilder().build()
