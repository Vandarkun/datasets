import json
import os
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

class UserHistoryLoader:
    def __init__(self, 
                 reviews_path, 
                 local_meta_path, 
                 min_reviews=5):
        """
        :param reviews_path: 原始 Amazon 评论文件 
        :param local_meta_path: 预处理好的元数据文件 
        :param min_reviews: 最小评论数
        """
        self.reviews_path = reviews_path
        self.local_meta_path = local_meta_path
        self.min_reviews = min_reviews
        self.meta_map = {} 
        self.user_groups = defaultdict(list)

    def _get_file_size_mb(self, path):
        try: return os.path.getsize(path) / (1024 * 1024)
        except: return 0

    def load_local_metadata(self):
        """步骤 1: 加载元数据到内存"""
        file_size = self._get_file_size_mb(self.local_meta_path)
        print(f"\n>> [Step 1/3] Loading Metadata from {os.path.basename(self.local_meta_path)} ({file_size:.2f} MB)...")
        
        count = 0
        with open(self.local_meta_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="   Reading Meta", unit=" items"):
                try:
                    item = json.loads(line.strip())
                    asin = item.get('asin')
                    if asin:
                        self.meta_map[asin] = item
                        count += 1
                except json.JSONDecodeError:
                    continue
        print(f"   Done. Loaded {count} metadata entries.")

    def process_reviews_and_group(self):
        """步骤 2: 读取评论 -> 过滤 -> 内存分组"""
        file_size = self._get_file_size_mb(self.reviews_path)
        print(f"\n>> [Step 2/3] Processing Reviews ({file_size:.2f} MB)...")
        
        kept = 0
        dropped = 0
        
        with open(self.reviews_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="   Filtering", unit=" reviews"):
                try:
                    review = json.loads(line.strip())
                    asin = review.get('asin')
                    uid = review.get('reviewerID')
                    
                    # 只有当 ASIN 存在元数据表中时，才保留这条评论
                    if uid and asin and (asin in self.meta_map):
                        compact_review = {
                            "asin": asin,
                            "time": review.get('unixReviewTime', 0),
                            "overall": float(review.get('overall', 0)),
                            "vote": review.get('vote', 0),
                            "summary": review.get('summary', ""),
                            "reviewText": review.get('reviewText', "")[:500]
                        }
                        
                        self.user_groups[uid].append(compact_review)
                        kept += 1
                    else:
                        dropped += 1
                        
                except json.JSONDecodeError:
                    continue
        
        print(f"   Kept: {kept} (Matched Meta), Dropped: {dropped}")

    def export_profiles(self, output_file):
        """步骤 3: 筛选用户并导出"""
        print(f"\n>> [Step 3/3] Generating Final Profiles...")
        
        initial_users = len(self.user_groups)
        valid_users = {
            uid: reviews 
            for uid, reviews in self.user_groups.items() 
            if len(reviews) >= self.min_reviews
        }
        print(f"   User Filtering: {initial_users} -> {len(valid_users)} (Min reviews: {self.min_reviews})")
        print(f"   Writing to: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for user_id, reviews in tqdm(valid_users.items(), desc="   Exporting", unit="user"):
                
                reviews.sort(key=lambda x: x['time'], reverse=True)
                
                user_history = []
                for r in reviews:
                    asin = r['asin']
                    meta = self.meta_map[asin] 
                    
                    votes = r['vote']
                    if isinstance(votes, str):
                        votes = int(votes.replace(',', '')) if votes.replace(',', '').isdigit() else 0

                    # 组装 Movie Meta
                    movie_meta_blob = {
                        "title": meta.get('title'),
                        "tmdb_id": meta.get('tmdb_id'),
                        "genres": meta.get('genres', []),
                        "director": meta.get('director', []),
                        "cast": meta.get('cast', []),
                        "keywords": meta.get('keywords', []),
                        "overview": meta.get('overview', ""),
                    }

                    # 组装交互记录
                    interaction = {
                        "timestamp": r['time'],
                        "date_str": datetime.fromtimestamp(r['time']).strftime('%Y-%m-%d'),
                        "rating": r['overall'],
                        "votes": votes,
                        "summary": r['summary'],
                        "review_text": r['reviewText'],
                        "asin": asin,
                        "movie_meta": movie_meta_blob
                    }
                    user_history.append(interaction)
                
                profile = {
                    "user_id": user_id,
                    "review_count": len(user_history),
                    "interaction_history": user_history
                }
                f_out.write(json.dumps(profile, ensure_ascii=False) + "\n")

        print(f"\n>> DONE! Saved to {output_file}")

    @staticmethod
    def read_first_n_jsonl_lines(file_path, output_path, n):
        """
        读取jsonl文件的前n行，解析为json格式并保存到新文件
        :param file_path: 输入的jsonl文件路径
        :param n: 要读取的行数
        :param output_path: 输出的json文件路径
        """
        result = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= n:
                    break
                try:
                    obj = json.loads(line.strip())
                    result.append(obj)
                except json.JSONDecodeError as e:
                    print(f"第{idx+1}行解析失败: {e}")
                    continue
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(result, out_f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    REVIEWS_FILE = "../data/Movies_and_TV.json" 
    LOCAL_META_FILE = "../data/tmdb_movie_metadata.jsonl" 
    OUTPUT_FILE = "../output/user_history_matched.jsonl"

    if os.path.exists(REVIEWS_FILE) and os.path.exists(LOCAL_META_FILE):
        loader = UserHistoryLoader(
            reviews_path=REVIEWS_FILE,
            local_meta_path=LOCAL_META_FILE,
            min_reviews=5 
        )
        
        loader.load_local_metadata()       
        loader.process_reviews_and_group()
        loader.export_profiles(OUTPUT_FILE)
        loader.read_first_n_jsonl_lines(OUTPUT_FILE, "../output/sample_user_history_matched.json", 5)
    else:
        print("Error: Input files not found.")

    
