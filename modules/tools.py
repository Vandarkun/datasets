import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
import config

# --- System 端工具: 电影检索 ---
class MovieRetriever:
    def __init__(self):
        print(f"Loading Embedding Model: {config.HF_MODEL_LOCAL_PATH}...")
        self.model = SentenceTransformer(config.HF_MODEL_LOCAL_PATH, device=config.EMBEDDING_DEVICE)
        
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
            
            query_vector = self.model.encode([keywords])
            
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
        初始化时传入用户的记忆列表。
        :param memories: List of dicts, e.g. [{'movie_title': '...', 'memory_text': '...'}]
        """
        self.memories = memories

    def lookup(self, query: str) -> str:
        """
        Search past movie memories to justify preferences.
        :param query: The movie topic or feeling to look for in memory.
        """
        hits = []
        query_lower = query.lower()
        
        # 简单的文本匹配逻辑
        for mem in self.memories:
            content = f"{mem.get('movie_title', '')} {mem.get('memory_text', '')}".lower()
            
            if query_lower in content:
                hits.append(
                    f"Film: {mem.get('movie_title')} | "
                    f"Rating: {mem.get('rating')} | "
                    f"Review: {mem.get('memory_text')}"
                )
        
        if not hits:
            return "No exact match in memory. I will rely on my general aesthetic preferences."
        
        return "\n".join(hits[:3])