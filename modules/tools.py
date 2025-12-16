import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer, util
import config

_SHARED_MODEL = None

def get_shared_model():
    global _SHARED_MODEL
    if _SHARED_MODEL is None:
        print(f"[Init] Loading Shared Embedding Model: {config.HF_MODEL_LOCAL_PATH}...")
        _SHARED_MODEL = SentenceTransformer(config.HF_MODEL_LOCAL_PATH, device=config.EMBEDDING_DEVICE)
    return _SHARED_MODEL

# --- System 端工具: 电影检索 ---
class MovieRetriever:
    def __init__(self):
        print(f"Loading Embedding Model: {config.HF_MODEL_LOCAL_PATH}...")
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
    def __init__(self, memories: list):
        """
        初始化时直接将所有记忆向量化并缓存。
        """
        self.model = get_shared_model()
        self.memories = memories
        self.memory_embeddings = None
        
        if self.memories:
            self.corpus = [
                f"{m.get('movie_title', '')}: {m.get('memory_text', '')}" 
                for m in memories
            ]
            print(f"[Init] Encoding {len(self.memories)} User Memories...")
            self.memory_embeddings = self.model.encode(self.corpus, convert_to_tensor=True, normalize_embeddings=True)

    def lookup(self, query: str) -> str:
        """
        Semantic search in user memories.
        """
        if not self.memories or self.memory_embeddings is None:
            return "My memory is blank (no history data)."

        try:
            # 向量化查询
            query_embedding = self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
            
            # 计算相似度
            cos_scores = util.cos_sim(query_embedding, self.memory_embeddings)[0]
            
            # 获取 Top 3
            top_results = list(zip(cos_scores, self.memories))
            top_results.sort(key=lambda x: x[0], reverse=True)
            
            hits = []
            # 阈值过滤
            threshold = 0.25
            
            for score, mem in top_results[:3]:
                if score < threshold:
                    continue
                hits.append(
                    f"[Similarity: {score:.2f}] Film: {mem.get('movie_title')} | "
                    f"Rating: {mem.get('rating')} | "
                    f"Review: {mem.get('memory_text')}"
                )
            
            if not hits:
                return "I racked my brain, but I can't recall any specific movie related to that topic."
            
            return "\n".join(hits)

        except Exception as e:
            return f"Memory Lookup Error: {e}"