import json
import os
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
import config
from modules.UserAgent import UserAgent
from modules.SystemAgent import SystemAgent

class DialogueController:
    def __init__(self, profile_path: str, output_path: str):
        self.profile_path = profile_path
        self.output_path = output_path
        
        self.user_profile = self._load_profile()
        
        print(f"--- Initializing Controller for User: {self.user_profile.get('user_id')} ---")
        self.user_agent = UserAgent(self.user_profile)
        self.system_agent = SystemAgent()
        
        self.judge_llm = ChatOpenAI(
            model=config.MODEL_NAME,
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
            temperature=0.0 
        )
        
        self.raw_log = [] 
        self.rejection_count = 0
        self.turn_count = 0
        self.is_finished = False

    def _load_profile(self) -> Dict:
        if not os.path.exists(self.profile_path):
            raise FileNotFoundError(f"Profile not found: {self.profile_path}")
        with open(self.profile_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _format_history(self, role_perspective: str) -> List[BaseMessage]:
        """
        为 LangChain 格式化历史消息。
        role_perspective: 'user' (User视角的history) 或 'system' (System视角的history)
        """
        history = []
        for entry in self.raw_log:
            content = entry['content']
            role = entry['role']
            
            if role_perspective == "user":
                if role == "user": history.append(AIMessage(content=content))
                else: history.append(HumanMessage(content=content))
            
            elif role_perspective == "system":
                if role == "system": history.append(AIMessage(content=content))
                else: history.append(HumanMessage(content=content))
        return history

    def _judge_intent(self, user_response: str) -> str:
        """调用 LLM 判断用户意图: ACCEPT, REJECT, or INQUIRY"""
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
            result = self.judge_llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
            if "ACCEPT" in result: return "ACCEPT"
            if "REJECT" in result: return "REJECT"
            return "INQUIRY"
        except Exception as e:
            print(f"Judge Error: {e}")
            return "INQUIRY"

    def run(self):
        """执行对话主循环"""
        # 1. System 开场
        init_msg = "Hi! I'm your movie assistant. How are you feeling today?"
        print(f"\n[System]: {init_msg}")
        self.raw_log.append({"role": "system", "content": init_msg})

        # 2. 循环交互
        while not self.is_finished:
            self.turn_count += 1
            
            # --- User Turn ---
            user_hist = self._format_history("user")
            last_sys_input = self.raw_log[-1]['content']
            
            # 这里的 rejection_count 会决定 User 是挑剔还是接受
            user_resp = self.user_agent.reply(last_sys_input, user_hist, self.rejection_count)
            
            print(f"\n[User]: {user_resp}")
            self.raw_log.append({"role": "user", "content": user_resp})

            # --- Judge Turn ---
            intent = self._judge_intent(user_resp)
            print(f"   >>> [Judge]: {intent} (Rejections: {self.rejection_count})")

            # 状态更新逻辑
            if intent == "ACCEPT":
                print("\n*** User Accepted. Conversation End. ***")
                self.is_finished = True
                break
            elif intent == "REJECT":
                self.rejection_count += 1
                print(f"   >>> Rejection count increased to {self.rejection_count}")

            # 轮次终止条件
            if self.turn_count >= config.MAX_TOTAL_TURNS:
                print("Force stop: Max turns reached.")
                break
            
            # 确保即使次数到了，也需要先由 System 发起最后一次推荐，User 才能在下一轮 Accept
            # (不需要额外逻辑，因为 agents.py 里只要 rejection_count >= MAX，User 下一次就会 Accept)

            # --- System Turn ---
            sys_hist = self._format_history("system")
            last_user_input = self.raw_log[-1]['content']
            
            sys_resp = self.system_agent.reply(last_user_input, sys_hist)
            print(f"\n[System]: {sys_resp}")
            self.raw_log.append({"role": "system", "content": sys_resp})

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
