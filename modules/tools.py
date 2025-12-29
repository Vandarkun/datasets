import faiss
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import config

_SHARED_MODEL = None

def get_shared_model():
    global _SHARED_MODEL
    if _SHARED_MODEL is None:
        print(f"[Init] Loading Embedding Model: {config.HF_MODEL_LOCAL_PATH}...")
        _SHARED_MODEL = SentenceTransformer(config.HF_MODEL_LOCAL_PATH, device=config.EMBEDDING_DEVICE)
    return _SHARED_MODEL


# --- System 端工具: 电影检索 ---
class MovieRetriever:
    def __init__(self):
        self.model = get_shared_model()
        
        print(f"Loading FAISS Index: {config.FAISS_INDEX_PATH}...")
        self.index = faiss.read_index(config.FAISS_INDEX_PATH)
        
        print(f"Loading Metadata: {config.FAISS_META_PATH}...")
        with open(config.FAISS_META_PATH, 'rb') as f:
            self.metadata = pickle.load(f)

    def search(self, keywords: str, exclude_titles: str = "") -> str:
        """
        Search for movies using semantic search.
        :param keywords: Description of the movie.
        :param exclude_titles: Comma-separated list of titles to ignore.
        """
        try:
            
            query_vector = self.model.encode([keywords], normalize_embeddings=True)
            
            if query_vector.dtype != 'float32':
                query_vector = query_vector.astype('float32')

            D, I = self.index.search(query_vector, 10)
            
            excludes = [t.strip().lower() for t in exclude_titles.split(",") if t.strip()]
            
            found_doc = None
            for idx in I[0]:
                if idx == -1: continue 
                
                meta = self.metadata[idx]['metadata']
                title = meta.get('title', 'Unknown').strip()
                
                if title.lower() in excludes:
                    continue
                
                found_doc = meta
                break
            
            if not found_doc:
                return "No suitable match found in database (all candidates might be excluded)."

            return (
                f"Best Match Found:\n"
                f"Title: {found_doc.get('title')}\n"
                f"Genres: {', '.join(found_doc.get('genres', []))}\n"
                f"Director: {', '.join(found_doc.get('director', []))}\n"
                f"Overview: {found_doc.get('overview', 'No overview')}\n"
            )
        except Exception as e:
            return f"Search Error: {e}"

# --- User 端工具: 记忆检索 (Class Refactored) ---
class MemoryRetriever:
    """
    离线记忆检索：基于预构建的 FAISS 索引。
    - 分别返回本用户与相似用户的命中，避免混淆。
    """

    def __init__(self, user_id: str, related_users: list[str], enable_related_memory: bool = True):
        self.model = get_shared_model()
        
        self.user_id = user_id
        self.enable_related_memory = enable_related_memory
        self.related_users = related_users if self.enable_related_memory else []
        self.index = None
        self.metadata = None
        self.user_index_map = {}
        self.threshold = 0.25
        self.top_k = 3
        self._load_index()

    def _load_index(self):
        idx_path = config.MEMORY_FAISS_INDEX_PATH
        meta_path = config.MEMORY_FAISS_META_PATH
        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            print(f"[MemoryRetriever] Memory index or meta not found: {idx_path}, {meta_path}")
            return
        try:
            print(f"Loading FAISS Index: {idx_path}...")
            self.index = faiss.read_index(idx_path)
            with open(meta_path, "rb") as f:
                print(f"Loading Metadata: {meta_path}...")
                self.metadata = pickle.load(f)
            # 为快速按 user 过滤，建立 user_id -> index 列表
            mapping = {}
            for i, meta in enumerate(self.metadata):
                uid = meta.get("user_id")
                if not uid:
                    continue
                mapping.setdefault(uid, []).append(i)
            self.user_index_map = mapping
            print(f"[MemoryRetriever] Loaded memory index with {self.index.ntotal} entries.")
        except Exception as e:
            print(f"[MemoryRetriever] Failed to load memory index: {e}")
            self.index = None
            self.metadata = None
            self.user_index_map = {}

    def _search_in_users(self, query_vec: np.ndarray, users: set[str]):
        if not self.index or not self.metadata or not users:
            return []
        # 收集这些用户的条目索引
        idxs = []
        for uid in users:
            idxs.extend(self.user_index_map.get(uid, []))
        if not idxs:
            return []
        # 重建向量并计算余弦（向量已归一化，安全起见再标准化一次）
        embeds = []
        metas = []
        for idx in idxs:
            try:
                vec = self.index.reconstruct(idx)
            except Exception:
                continue
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeds.append(vec)
            metas.append(self.metadata[idx])
        if not embeds:
            return []
        mat = np.vstack(embeds)
        scores = mat @ query_vec.T  # (n,1)
        scores = scores.reshape(-1)

        pairs = []
        for score, meta in zip(scores, metas):
            if score < self.threshold:
                continue
            pairs.append((float(score), meta))
        pairs.sort(key=lambda x: x[0], reverse=True)
        return pairs[: self.top_k]

    def lookup(self, query: str) -> str:
        """
        Semantic search in prebuilt memory index.
        Returns sections for self and related users separately.
        """
        if not self.index or not self.metadata:
            return "Memory index not available. Please build user memory FAISS first."
        try:
            query_vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            if query_vec.dtype != np.float32:
                query_vec = query_vec.astype("float32")

            self_hits = self._search_in_users(query_vec, {self.user_id})
            rel_hits = []
            if self.enable_related_memory and self.related_users:
                rel_hits = self._search_in_users(query_vec, set(self.related_users))
            similar_disabled = not self.enable_related_memory

            def fmt(hits, label, disabled=False):
                if disabled:
                    return f"{label}: disabled by config."
                if not hits:
                    return f"{label}: no relevant memories."
                lines = []
                for score, mem in hits:
                    lines.append(
                        f"[{label} | Sim {score:.2f}] User: {mem.get('user_id')} | "
                        f"Film: {mem.get('movie_title')} | Rating: {mem.get('rating')} | "
                        f"Memory: {mem.get('memory_text')}"
                    )
                return "\n".join(lines)

            sections = []
            sections.append(fmt(self_hits, "Your Memory"))
            sections.append(fmt(rel_hits, "Similar User Memory", disabled=similar_disabled))
            return "\n".join(sections)

        except Exception as e:
            return f"Memory Lookup Error: {e}"
