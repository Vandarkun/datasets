import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from openai import OpenAI
from tqdm import tqdm


class KeyMemoryItem(BaseModel):
    movie_title: str
    rating: float
    memory_text: str = Field(..., description="A concise summary of their specific opinion/feeling.")

class KeyMemoryList(BaseModel):
    memories: List[KeyMemoryItem]

class ReflectionProfile(BaseModel):
    aesthetic_preferences: List[str] = Field(..., description="What elements do they love/hate?")
    spectator_persona: str = Field(..., description="Label like 'Critical Historian' or 'Popcorn Fan'")
    decision_logic: str = Field(..., description="Why they choose movies?")
    taste_evolution: str = Field(..., description="How their taste changed over time. e.g. 'Started loving Horror but shifted to Family films in 2014'.")
    contradictions: Optional[str] = Field(None, description="Any conflicting tastes?")

class StyleProfile(BaseModel):
    tone: str
    verbosity: str
    common_keywords: List[str]
    review_structure: str

class FullUserProfile(BaseModel):
    user_id: str
    meta_stats: dict
    key_memories: List[KeyMemoryItem]
    reflections: ReflectionProfile
    dialogue_style: StyleProfile
    related_users: List[str] = Field(default_factory=list, description="Top related user IDs for memory expansion.")


def load_neighbor_map(neighbor_file: Optional[str]) -> Dict[str, List[str]]:
    """
    Load neighbor user IDs from jsonl produced by build_social_graph.py.
    Each line: {"user_id": "...", "neighbors": [{"user_id": "...", ...}, ...]}
    """
    if not neighbor_file or not os.path.exists(neighbor_file):
        return {}
    mapping: Dict[str, List[str]] = {}
    with open(neighbor_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            uid = obj.get("user_id")
            neighbors = obj.get("neighbors", [])
            if not uid:
                continue
            mapping[uid] = [n.get("user_id") for n in neighbors if n.get("user_id")]
    return mapping


class MemoryProfileChain:
    def __init__(self, api_key, base_url, model_name="deepseek-chat", neighbor_file: Optional[str] = None, neighbor_map: Optional[Dict[str, List[str]]] = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.neighbor_map = neighbor_map or load_neighbor_map(neighbor_file)

    def _time_aware_sample(self, interaction_history):
        """
        [Core Logic] 时序感知采样策略
        将历史划分为 Early/Middle/Recent，确保采样覆盖用户生命周期
        """
        if not interaction_history: return []
        
        valid_reviews = [r for r in interaction_history if len(r.get('review_text', '')) > 30]
        if not valid_reviews: return []
        
        sorted_history = sorted(valid_reviews, key=lambda x: x['timestamp'])
        total = len(sorted_history)
        
        if total < 5:
            return sorted_history 

        early_end = max(1, int(total * 0.2))          # 前 20%
        recent_start = min(total - 1, int(total * 0.8)) # 后 20%

        early_era = sorted_history[:early_end]
        middle_era = sorted_history[early_end:recent_start]
        recent_era = sorted_history[recent_start:]
        
        final_samples = []

        # [A] 早期 (The Origin): 确立初始人设 (1 High + 1 Low)
        if early_era:
            final_samples.append(max(early_era, key=lambda x: x['rating'])) 
            early_lows = [r for r in early_era if r['rating'] <= 3.0]
            if early_lows: 
                final_samples.append(min(early_lows, key=lambda x: x['rating']))

        # [B] 中期 (The Evolution): 寻找深度交互 (字数最长)
        if middle_era:
            longest = max(middle_era, key=lambda x: len(x['review_text']))
            final_samples.append(longest)
            # 补一个反例来增加多样性
            if longest['rating'] > 3:
                mid_lows = [r for r in middle_era if r['rating'] <= 3]
                if mid_lows: final_samples.append(mid_lows[0])
            else:
                mid_highs = [r for r in middle_era if r['rating'] > 3]
                if mid_highs: final_samples.append(mid_highs[0])

        # [C] 近期 (The Now): 严格取最后 3 条，反映当下意图
        final_samples.extend(recent_era[-3:])

        # 4. 去重
        seen_asins = set()
        unique_samples = []
        for item in final_samples:
            if item['asin'] not in seen_asins:
                unique_samples.append(item)
                seen_asins.add(item['asin'])
        
        unique_samples.sort(key=lambda x: x['timestamp'])
        
        return unique_samples

    def _format_input_text(self, sampled_reviews):
        text_block = ""
        for r in sampled_reviews:
            meta = r.get('movie_meta', {})
            date_str = r.get('date_str', 'Unknown Date')
            
            text_block += f"--- Date: {date_str} ---\n"
            text_block += f"Movie: {meta.get('title', 'Unknown')} ({meta.get('release_year', '')})\n"
            text_block += f"Director: {', '.join(meta.get('director', []))}\n"
            text_block += f"Rating: {r['rating']}/5.0\n"
            text_block += f"Review: {r['review_text']}\n\n"
        return text_block

    def _call_llm(self, system_prompt, user_content, pydantic_model):
        schema = json.dumps(pydantic_model.model_json_schema(), indent=2)
        full_system_prompt = f"{system_prompt}\n\nIMPORTANT: Output valid JSON only following this schema:\n{schema}"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            content = response.choices[0].message.content
            return pydantic_model.model_validate(json.loads(content))
        except Exception as e:
            print(f"LLM Call Failed: {e}")
            return None

    # Step 1: 提取记忆
    def _step_1_memories(self, context_text):
        # print("    > Step 1: Extracting Memories...")
        prompt = "Analyze reviews. Extract concise key memories. Focus on WHY they liked/disliked specific movies."
        result = self._call_llm(prompt, context_text, KeyMemoryList)
        return result.memories if result else []

    # Step 2: 深度反思 (含演变分析)
    def _step_2_reflections(self, context_text, memories, date_range_str):
        # print("    > Step 2: Synthesizing Reflections & Evolution...")
        memory_summary = "\n".join([f"- {m.movie_title}: {m.memory_text}" for m in memories])
        
        # Prompt 引导分析时间演变
        prompt = f"""
        You are a psychologist profiling a movie viewer based on their review history ({date_range_str}).
        
        Summarized Memories:
        {memory_summary}
        
        Task:
        1. **Identify Taste Evolution**: Did their taste change over the years? (e.g. Action -> Family).
        2. **Identify Stable Core**: What stayed the same?
        3. **Reflect on Context**: Infer life changes based on the timeline.
        """
        return self._call_llm(prompt, context_text, ReflectionProfile)

    # Step 3: 风格分析
    def _step_3_style(self, context_text):
        # print("    > Step 3: Analyzing Style...")
        prompt = "Analyze the writing style (tone, verbosity, keywords) to help a chatbot mimic this user."
        return self._call_llm(prompt, context_text, StyleProfile)

    def process_user(self, user_data):
        user_id = user_data['user_id']
        raw_history = user_data.get('interaction_history', [])
        
        sampled = self._time_aware_sample(raw_history)
        if not sampled: 
            print("No valid samples found.")
            return None
            
        context = self._format_input_text(sampled)
        
        start_date = sampled[0].get('date_str', 'Unknown')
        end_date = sampled[-1].get('date_str', 'Unknown')
        date_range = f"{start_date} to {end_date}"
        
        try:
            memories = self._step_1_memories(context)
            if not memories: return None
            
            reflections = self._step_2_reflections(context, memories, date_range)
            style = self._step_3_style(context)
            
            related_users = self.neighbor_map.get(user_id, [])

            return FullUserProfile(
                user_id=user_id,
                meta_stats={
                    "total_reviews": len(raw_history), 
                    "sampled": len(sampled),
                    "time_span": date_range
                },
                key_memories=memories,
                reflections=reflections,
                dialogue_style=style,
                related_users=related_users
            ).model_dump()
            
        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
            return None

if __name__ == "__main__":
    
    INPUT_FILE = "../output/user_history_matched.jsonl" 
    OUTPUT_FILE = "../output/sample_profile_100.json"
    NEIGHBOR_FILE = "/data/wdk/datasets/output/user_neighbors.jsonl"
    N = 100  # 0 表示处理所有用户, 否则仅处理前 N 个用户
    MAX_WORKERS = 8  # 并行线程数

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_users = [json.loads(line.strip()) for line in f if line.strip()]
        selected_users = all_users if N <= 0 else all_users[:N]
        print(f"Loaded {len(all_users)} users, processing {len(selected_users)} (N={N}).")
    
    profiler = MemoryProfileChain(api_key="sk-fd424ac8b68d4e3fbe0dc9988ff4cc65",
                                base_url="https://api.deepseek.com",
                                model_name="deepseek-chat",
                                neighbor_file=NEIGHBOR_FILE)
    
    indexed_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(profiler.process_user, user): idx for idx, user in enumerate(selected_users)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="处理用户"):
            r = fut.result()
            if r:
                indexed_results.append((futures[fut], r))

    indexed_results.sort(key=lambda x: x[0])
    results = [r for _, r in indexed_results]
    
    if results:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(results)} 个用户到 {OUTPUT_FILE}")
