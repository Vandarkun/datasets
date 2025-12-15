import csv
import json
import requests
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# ================= 配置区 =================
API_KEY = "0502195f6ecb8fa1d60ac1fe46b4f2e0"

# 文件路径
INPUT_CSV = 'D:\\Desktop\\My\\code\\datasets\\data\\title_list.csv'
OUTPUT_FILE = 'D:\\Desktop\\My\\code\\datasets\\data\\tmdb_movie_metadata.jsonl'

# 并发线程数
MAX_WORKERS = 5
# ==========================================

write_lock = threading.Lock()

def create_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

session = create_session()

def search_movie_id(title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {'api_key': API_KEY, 'query': title, 'language': 'en-US', 'page': 1}
    try:
        resp = session.get(url, params=params, timeout=10)
        if resp.status_code == 429:
            time.sleep(2)
            return search_movie_id(title)
        if resp.status_code == 200:
            results = resp.json().get('results', [])
            if results:
                return results[0]['id'], results[0]['release_date']
    except Exception as e:
        print(f"Error searching {title}: {e}")
    return None, None

def get_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {'api_key': API_KEY, 'append_to_response': 'credits,keywords', 'language': 'en-US'}
    try:
        resp = session.get(url, params=params, timeout=10)
        if resp.status_code == 429:
            time.sleep(2)
            return get_movie_details(movie_id)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Error details {movie_id}: {e}")
    return None

def process_single_row(row):
    asin = row['asin']
    clean_title = row['clean_title']
    
    tmdb_id, release_date = search_movie_id(clean_title)
    if not tmdb_id: return None 

    details = get_movie_details(tmdb_id)
    if not details: return None

    try:
        # 提取字段
        crew = details.get('credits', {}).get('crew', [])
        directors = [m['name'] for m in crew if m['job'] == 'Director']
        
        cast = details.get('credits', {}).get('cast', [])
        top_cast = [m['name'] for m in cast[:5]]
        
        keywords = [k['name'] for k in details.get('keywords', {}).get('keywords', [])]
        genres = [g['name'] for g in details.get('genres', [])]
        overview = details.get('overview', "") # 这里的 "" 保证了 overview 即使是 None 也会变成空字符串

        # ================= 修改后的筛选逻辑 =================
        # 逻辑：只有当所有关键信息全是空的，才认为是废数据
        # 只要有其中任意一个有值，就保留
        
        is_empty_overview = (not overview)
        is_empty_genres = (not genres)       # 空列表 [] 也是 False
        is_empty_cast = (not top_cast)       # 空列表 [] 也是 False
        is_empty_keywords = (not keywords)   # 空列表 [] 也是 False

        # 如果全部都为空，则返回 None (丢弃)
        if is_empty_overview and is_empty_genres and is_empty_cast and is_empty_keywords:
            # 可以在这里打印一下，看看哪些被丢了
            # print(f"丢弃全空数据: {clean_title}") 
            return None
        # ===================================================

        meta_data = {
            "asin": asin,
            "tmdb_id": tmdb_id,
            "title": details.get('title'),
            "overview": overview,
            "genres": genres,
            "director": directors,
            "cast": top_cast,
            "keywords": keywords,
        }
        return meta_data
        
    except Exception as e:
        print(f"Error parsing {clean_title}: {e}")
        return None

def main():
    processed_asins = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_asins.add(data['asin'])
                except: pass
    print(f"🔄 跳过 {len(processed_asins)} 条已有数据")

    if not os.path.exists(INPUT_CSV):
        print(f"❌ 找不到输入文件: {INPUT_CSV}")
        return

    tasks = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['asin'] not in processed_asins:
                tasks.append(row)

    print(f"🚀 开始多线程处理 {len(tasks)} 条数据 (线程数: {MAX_WORKERS})")
    print("⚡ 按 Ctrl+C 可随时安全停止")

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    future_to_row = {}
    f_out = open(OUTPUT_FILE, 'a', encoding='utf-8')

    try:
        for row in tasks:
            future = executor.submit(process_single_row, row)
            future_to_row[future] = row

        for future in tqdm(as_completed(future_to_row), total=len(tasks)):
            result = future.result()
            if result:
                with write_lock:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()

    except KeyboardInterrupt:
        print("\n🛑 正在停止... 取消剩余任务中...")
        executor.shutdown(wait=False)
        print("✅ 已安全退出。数据已保存。")
    
    finally:
        f_out.close()

if __name__ == "__main__":
    if "YOUR_NEW" in API_KEY:
        print("❌ 请先修改代码填入新的 API Key！")
    else:
        main()