import json
import os
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

MODEL_PATH = '../model/all-MiniLM-L6-v2' 
# MAX_INTERACTIONS_PER_USER = 120  

def load_users(path: str, max_users: int | None = None) -> List[dict]:
    """Load user histories from jsonl or json list."""
    users: List[dict] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"User history file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file should contain a list of users.")
            for item in data:
                users.append(item)
                if max_users and len(users) >= max_users:
                    break
        else:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                users.append(obj)
                if max_users and len(users) >= max_users:
                    break
    return users


def _compose_review_text(interaction: dict) -> Tuple[str, float, List[str]]:
    """Create a text snippet and weight for one interaction."""
    meta = interaction.get("movie_meta", {})
    title = meta.get("title", "Unknown Title")
    genres = meta.get("genres", []) or []
    rating = float(interaction.get("rating", 0) or 0)
    summary = interaction.get("summary", "")
    review_text = interaction.get("review_text", "") or summary

    # 基于评分的权重: 评分越高 => 权重越高; 但最低为0.3
    weight = max(0.3, min(1.5, 0.5 + rating / 5.0))

    text = (
        f"{title}. Genres: {', '.join(genres)}. "
        f"Rating: {rating:.1f}/5. "
        f"Summary: {summary}. Review: {review_text}"
    )
    return text, weight, genres


def build_user_embeddings(
    users: List[dict],
    min_reviews: int = 8,
    batch_size: int = 64,
) -> Tuple[np.ndarray, List[Dict]]:
    """Return (embeddings array, metadata list) for users with enough reviews."""
    model = SentenceTransformer(MODEL_PATH, device="cuda")
    user_embeddings: List[np.ndarray] = []
    user_meta: List[Dict] = []

    for user in tqdm(users, desc="编码用户向量"):
        interactions = user.get("interaction_history", [])
        if len(interactions) < min_reviews:
            continue

        interactions = sorted(
            interactions,
            key=lambda x: x.get("timestamp", 0),
            reverse=True
        )

        # if MAX_INTERACTIONS_PER_USER:
        #     interactions = interactions[:MAX_INTERACTIONS_PER_USER]

        snippets: List[str] = []
        weights: List[float] = []
        genres_set = set()

        for inter in interactions:
            text, weight, genres = _compose_review_text(inter)
            snippets.append(text)
            weights.append(weight)
            genres_set.update([g for g in genres if g])

        if not snippets:
            continue

        vecs = model.encode(
            snippets,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        w = np.array(weights, dtype=np.float32).reshape(-1, 1)
        pooled = (vecs * w).sum(axis=0)
        norm = np.linalg.norm(pooled)
        if norm == 0:
            pooled = vecs.mean(axis=0)
            norm = np.linalg.norm(pooled)
        user_vec = (pooled / norm).astype("float32")

        user_embeddings.append(user_vec)
        user_meta.append(
            {
                "user_id": user.get("user_id"),
                "genres": genres_set,
                "review_count": len(interactions),
            }
        )

    if not user_embeddings:
        raise ValueError("No valid users were encoded (check min_reviews or input data).")

    return np.stack(user_embeddings), user_meta


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create a cosine-similarity index (use normalized vectors)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def find_neighbors(
    index: faiss.IndexFlatIP,
    embeddings: np.ndarray,
    metas: List[Dict],
    top_k: int = 5,
    min_sim: float = 0.25,
    max_shared_genres: int = 5,
) -> List[Dict]:
    """Return neighbors for each user."""
    search_k = top_k + 1  # 跳过本身
    scores, indices = index.search(embeddings, search_k)
    results: List[Dict] = []

    for i, (row_scores, row_indices) in tqdm(
        enumerate(zip(scores, indices)),
        total=len(metas),
        desc="查找邻居",
    ):
        base_meta = metas[i]
        neighbors = []
        for sim_idx, sim_score in zip(row_indices, row_scores):
            if sim_idx == i:
                continue  # 跳过本身
            if sim_score < min_sim:
                continue

            n_meta = metas[sim_idx]
            shared = sorted(list(base_meta["genres"].intersection(n_meta["genres"])))
            neighbors.append(
                {
                    "user_id": n_meta["user_id"],
                    "score": float(sim_score),
                    "shared_genres": shared[:max_shared_genres],
                    "shared_genre_count": len(shared),
                    "neighbor_review_count": n_meta["review_count"],
                }
            )
            if len(neighbors) >= top_k:
                break

        results.append({"user_id": base_meta["user_id"], "neighbors": neighbors})
    return results


def save_neighbors(neighbors: List[Dict], output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in neighbors:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已将邻居图保存到 {output_path}（用户数：{len(neighbors)}）")

CONFIG = {
    "input": "/data/wdk/datasets/output/user_history_matched.jsonl",
    "output": "/data/wdk/datasets/output/user_neighbors.jsonl",
    "top_k": 5,
    "min_reviews": 8,
    "min_sim": 0.25,
    "max_users": None,
}


def main():
    users = load_users(CONFIG["input"], max_users=CONFIG["max_users"])
    print(f"已从 {CONFIG['input']} 加载 {len(users)} 个用户")
    print(f"开始编码用户向量（最少影评：{CONFIG['min_reviews']}）")

    embeddings, metas = build_user_embeddings(
        users,
        min_reviews=CONFIG["min_reviews"],
    )
    print(f"编码完成，有效用户数：{len(metas)}")
    print("开始构建FAISS索引")
    index = build_faiss_index(embeddings)
    print(f"开始查找邻居（top_k={CONFIG['top_k']}，min_sim={CONFIG['min_sim']}）")
    neighbors = find_neighbors(
        index,
        embeddings,
        metas,
        top_k=CONFIG["top_k"],
        min_sim=CONFIG["min_sim"],
    )
    print(f"开始保存邻居结果到 {CONFIG['output']}")
    save_neighbors(neighbors, CONFIG["output"])


if __name__ == "__main__":
    main()
