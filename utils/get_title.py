import json
import csv
import re
import os
from tqdm import tqdm 

INPUT_FILE = 'D:\\Desktop\\My\\code\\datasets\\data\\meta_Movies_and_TV.json'
OUTPUT_FILE = 'D:\\Desktop\\My\\code\\datasets\\data\\title_list.csv'

# --- 2. 清洗函数  ---
def clean_amazon_title(title):
    if not title: return ""
    title = re.sub(r"\[.*?\]", "", title)
    title = re.sub(r"\(.*?\)", "", title)
    return title.strip()

# --- 3. 主处理函数 ---
def extract_movies():
    # 检查文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        print("请确认文件名是否正确，或者文件是否在当前目录下。")
        return

    print(f"正在读取 {INPUT_FILE} ...")
    
    count = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(['asin', 'raw_title', 'clean_title'])

        # --- 【关键修改】直接使用 open ---
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                try:
                    item = json.loads(line)
                    
                    asin = item.get('asin')
                    title = item.get('title')
                    
                    if asin and title:
                        clean_t = clean_amazon_title(title)
                        writer.writerow([asin, title, clean_t])
                        count += 1
                        
                except json.JSONDecodeError:
                    continue

    print(f"\n✅ 提取完成！共提取了 {count} 部电影。")
    print(f"📁 结果已保存至: {OUTPUT_FILE}")

# --- 4. 运行 ---
if __name__ == "__main__":
    extract_movies()