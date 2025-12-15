from langchain_openai import ChatOpenAI

API_KEY = "sk-fd424ac8b68d4e3fbe0dc9988ff4cc65" 
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

MAX_REJECTIONS = 2
MAX_TOTAL_TURNS = 12

FAISS_INDEX_PATH = r"D:\Desktop\My\code\datasets\data\faiss"

HF_MODEL_LOCAL_PATH = r"D:\Desktop\My\code\datasets\model\all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

class BaseAgent:
    def __init__(self, temperature=0.2):
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=temperature,
            max_tokens=8192
        )

# MOVIE_KNOWLEDGE_BASE = [
#     {
#         "title": "The Secret Life of Walter Mitty",
#         "year": 2013,
#         "genre": "Adventure/Drama",
#         "tags": ["uplifting", "visuals", "self-discovery", "travel"],
#         "desc": "A day-dreamer escapes his anonymous life by disappearing into a world of fantasies filled with heroism, romance and action."
#     },
#     {
#         "title": "Taken",
#         "year": 2008,
#         "genre": "Action/Thriller",
#         "tags": ["revenge", "fast-paced", "violence", "kidnapping"],
#         "desc": "A retired CIA agent travels across Europe and relies on his old skills to save his estranged daughter."
#     },
#     {
#         "title": "Paddington 2",
#         "year": 2017,
#         "genre": "Family/Comedy",
#         "tags": ["feel-good", "kindness", "humor", "wholesome"],
#         "desc": "Paddington, now happily settled with the Brown family, picks up odd jobs to buy a gift for his aunt."
#     },
#     {
#         "title": "Prisoners",
#         "year": 2013,
#         "genre": "Thriller/Crime",
#         "tags": ["dark", "intense", "suspense", "moral ambiguity"],
#         "desc": "When his daughter goes missing, a desperate father takes matters into his own hands."
#     },
#     {
#         "title": "Silver Linings Playbook",
#         "year": 2012,
#         "genre": "Drama/Romance",
#         "tags": ["redemption", "mental health", "acting", "dramedy"],
#         "desc": "After a stint in a mental institution, former teacher Pat Solitano moves back in with his parents."
#     }
# ]