import json
import os
from openai import OpenAI
import config
from modules.UserAgent import UserAgent
from modules.SystemAgent import SystemAgent

class DialogueController:
    def __init__(self, profile_path: str, output_path: str):
        self.profile_path = profile_path
        self.output_path = output_path
        self.user_profile = self._load_profile()
        
        print(f"--- Initializing AutoGen Controller for User: {self.user_profile.get('user_id')} ---")
        
        # 初始化两个 AutoGen 封装 Agent
        self.user_agent = UserAgent(self.user_profile)
        self.system_agent = SystemAgent()
        
        # 初始化 Judge (直接用 OpenAI Client 即可，轻量级)
        self.client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
        
        self.raw_log = [] 
        self.rejection_count = 0
        self.turn_count = 0
        self.is_finished = False

    def _load_profile(self) -> dict:
        if not os.path.exists(self.profile_path):
            raise FileNotFoundError(f"Profile not found: {self.profile_path}")
        with open(self.profile_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _judge_intent(self, user_response: str) -> str:
        """简单的意图判断"""
        prompt = f"""
        Analyze the user response in a movie recommendation context.
        User Response: "{user_response}"
        
        Classify into EXACTLY one category:
        1. ACCEPT: User agrees to watch the movie.
        2. REJECT: User expresses disinterest or dislike.
        3. INQUIRY: User is answering a question, chatting, or asking info (neutral).
        
        Output ONLY the category word.
        """
        try:
            resp = self.client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            result = resp.choices[0].message.content.strip().upper()
            
            if "ACCEPT" in result: return "ACCEPT"
            if "REJECT" in result: return "REJECT"
            return "INQUIRY"
        except Exception as e:
            print(f"Judge Error: {e}")
            return "INQUIRY"

    def  run(self):
        # System 开场
        init_msg = "Hi! I'm your movie assistant. How are you feeling today?"
        print(f"\n[System]: {init_msg}")
        self.raw_log.append({"role": "system", "content": init_msg})
        
        last_msg = init_msg

        # 2. 循环交互
        while not self.is_finished:
            self.turn_count += 1
            
            # --- User Turn ---
            # 直接调用 UserAgent 的 reply 方法，它内部会运行 AutoGen 的 loop
            user_resp = self.user_agent.reply(last_msg, self.raw_log, self.rejection_count)
            
            print(f"\n[User]: {user_resp}")
            self.raw_log.append({"role": "user", "content": user_resp})

            # --- Judge Turn ---
            intent = self._judge_intent(user_resp)
            print(f"   >>> [Judge]: {intent} (Rejections: {self.rejection_count})")

            if intent == "ACCEPT":
                print("\n*** User Accepted. Conversation End. ***")
                self.is_finished = True
                break
            elif intent == "REJECT":
                self.rejection_count += 1

            if self.turn_count >= config.MAX_TOTAL_TURNS:
                print("Force stop: Max turns reached.")
                break

            # --- System Turn ---
            sys_resp = self.system_agent.reply(user_resp, self.raw_log)
            print(f"\n[System]: {sys_resp}")
            self.raw_log.append({"role": "system", "content": sys_resp})
            last_msg = sys_resp

        self.save_results()

    def save_results(self):
        data = {
            "user_id": self.user_profile.get("user_id"),
            "meta_stats": self.user_profile.get("meta_stats"),
            "final_rejection_count": self.rejection_count,
            "turns": self.turn_count,
            "dialogue": self.raw_log
        }
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nData successfully saved to {self.output_path}")