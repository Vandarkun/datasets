import json
import os
from openai import OpenAI
import config
from modules.UserAgent import UserAgent
from modules.SystemAgent import SystemAgent

def print_section(title, char="=", length=60):
    print(f"\n{char * length}")
    print(f"  {title}")
    print(f"{char * length}")

def print_final_response(role, content):
    border = "-" * 60
    print(f"\n{border}")
    print(f" [{role.upper()} FINAL RESPONSE]:")
    print(f"{content}")
    print(f"{border}\n")

class DialogueController:
    def __init__(self, profile_path: str, output_path: str):
        self.profile_path = profile_path
        self.output_path = output_path
        
        dir_name = os.path.dirname(output_path)
        base_name = os.path.basename(output_path).replace(".json", "")
        self.script_path = os.path.join(dir_name, "logs", f"{base_name}_script.txt")
        
        self.user_profile = self._load_profile()
        
        print(f"--- Initializing AutoGen Controller for User: {self.user_profile.get('user_id')} ---")
        
        self.user_agent = UserAgent(self.user_profile)
        self.system_agent = SystemAgent()
        
        self.client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
        
        self.raw_log = [] 
        self.rejection_count = 0
        self.turn_count = 0
        self.is_finished = False
        
        with open(self.script_path, "w", encoding="utf-8") as f:
            f.write(f"=== Dialogue Script for User: {self.user_profile.get('user_id')} ===\n\n")

    def _load_profile(self) -> dict:
        if not os.path.exists(self.profile_path):
            raise FileNotFoundError(f"Profile not found: {self.profile_path}")
        with open(self.profile_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _log_to_script(self, role, content, extra_info=""):
        with open(self.script_path, "a", encoding="utf-8") as f:
            timestamp = f"[Turn {self.turn_count}]" if self.turn_count > 0 else "[Init]"
            header = f"{timestamp} {role}: {extra_info}"
            f.write(f"{header}\n{content}\n\n{'-'*30}\n\n")

    def _judge_intent(self, user_response: str) -> str:
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
        except Exception:
            return "INQUIRY"

    def run(self):
        init_msg = "Hi! I'm your movie assistant. How are you feeling today?"
        
        print_final_response("SYSTEM", init_msg)
        self.raw_log.append({"role": "system", "content": init_msg})
        self._log_to_script("System", init_msg)
        
        last_msg = init_msg

        while not self.is_finished:
            self.turn_count += 1
            
            print_section(f"ROUND {self.turn_count} START", char="=")

            # --- User Turn ---
            print_section(f"USER TURN (Thinking & Memory Search...)", char="-")
            user_resp = self.user_agent.reply(last_msg, self.raw_log, self.rejection_count)
            
            print_final_response("USER", user_resp)
            self.raw_log.append({"role": "user", "content": user_resp})
            self._log_to_script("User", user_resp)

            # --- Judge Turn ---
            intent = self._judge_intent(user_resp)
            print(f"    [JUDGE]: {intent} (Rejections: {self.rejection_count})")

            # 状态更新
            if intent == "ACCEPT":
                print("\n *** User Accepted. Conversation End. ***")
                self._log_to_script("Judge", "User Accepted -> END")
                self.is_finished = True
                break
            elif intent == "REJECT":
                self.rejection_count += 1
                self._log_to_script("Judge", f"User Rejected (Count: {self.rejection_count})")

            if self.turn_count >= config.MAX_TOTAL_TURNS:
                print(" Force stop: Max turns reached.")
                self._log_to_script("Judge", "Max turns reached -> END")
                break

            # --- System Turn ---
            print_section(f"SYSTEM TURN (Thinking & Database Search...)", char="-")
            sys_resp = self.system_agent.reply(user_resp, self.raw_log)
            
            print_final_response("SYSTEM", sys_resp)
            self.raw_log.append({"role": "system", "content": sys_resp})
            self._log_to_script("System", sys_resp)
            
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
        print(f"\nData saved to {self.output_path}")
        print(f" Clean script saved to {self.script_path}")