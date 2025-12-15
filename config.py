import os

API_KEY = "sk-fd424ac8b68d4e3fbe0dc9988ff4cc65"  
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

LLM_CONFIG = {
    "config_list": [
        {
            "model": MODEL_NAME,
            "api_key": API_KEY,
            "base_url": BASE_URL,
        }
    ],
    "temperature": 0.2,
    "timeout": 120,
}

MAX_REJECTIONS = 2
MAX_TOTAL_TURNS = 8

FAISS_INDEX_PATH = r"D:\Desktop\My\code\datasets\data\faiss\movies.faiss"
FAISS_META_PATH = r"D:\Desktop\My\code\datasets\data\faiss\movies.pkl"
HF_MODEL_LOCAL_PATH = r"D:\Desktop\My\code\datasets\model\all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"