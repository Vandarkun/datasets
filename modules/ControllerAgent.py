import json
import os
import re
from openai import OpenAI
from typing import Tuple
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
    def __init__(
        self,
        profile_path: str = "",
        output_path: str = "",
        profile_data: dict | None = None,
        enable_result_file: bool = True,
    ):
        self.profile_path = profile_path
        self.output_path = output_path
        self.enable_result_file = enable_result_file
        
        self.user_profile = profile_data if profile_data is not None else self._load_profile()
        
        print(f"--- Initializing AutoGen Controller for User: {self.user_profile.get('user_id')} ---")
        
        self.user_agent = UserAgent(self.user_profile)
        self.system_agent = SystemAgent()
        
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
    
    def _review_user_response(self, user_response: str) -> tuple[bool, str]:
        """
        审核 UserAgent 生成的回复是否符合 PROFILE
        返回: (是否符合, 反馈信息)
        """
        reflections = self.user_profile.get("reflections", {})
        style = self.user_profile.get("dialogue_style", {})
        
        persona = reflections.get("spectator_persona", "")
        tone = style.get("tone", "")
        preferences = reflections.get("aesthetic_preferences", [])
        verbosity = style.get("verbosity", "") 
        decision_logic = reflections.get("decision_logic", "")
        
        prompt = f"""
        You are a reviewer responsible for checking if a user's response matches their personal profile (PROFILE).
        
        **User PROFILE Information:**
        - Persona: {persona}
        - Tone: {tone}
        - Preferences: {json.dumps(preferences, ensure_ascii=False)}
        - Verbosity: {verbosity}
        - Decision Logic: {decision_logic}
        
        **User's Generated Response:**
        "{user_response}"
        
        **Review Task:**
        Please check if the user's response matches their PROFILE. Pay special attention to:
        1. Whether the response's tone matches the tone specified in the PROFILE
        2. Whether the response's content reflects the persona in the PROFILE
        3. Whether the response's expression style matches the verbosity description
        4. Whether the response reflects the preferences and decision_logic in the PROFILE
        
        **Output Format:**
        If it matches the PROFILE, output: PASS
        If it does not match the PROFILE, output: FAIL
        
        If it does not match, please briefly explain the specific reason in one line (e.g., tone is too formal and does not match conversational style, content does not reflect persona characteristics, etc.).
        Format: FAIL|reason explanation
        
        **Output:**
        """
        
        try:
            resp = self.client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            result = resp.choices[0].message.content.strip()
            
            if result.upper().startswith("PASS"):
                return True, ""
            elif result.upper().startswith("FAIL"):
                # 提取原因说明
                parts = result.split("|", 1)
                reason = parts[1].strip() if len(parts) > 1 else "Response does not match PROFILE requirements"
                return False, reason
            else:
                # 如果输出格式不符合预期，默认为通过（避免过于严格）
                return True, ""
        except Exception as e:
            print(f"    [REVIEW ERROR]: {e}")
            # 出错时默认通过，避免阻塞流程
            return True, ""

    def _review_coherence(self, user_response: str) -> Tuple[bool, str]:
        """
        审查对话连贯性：检查回复是否与对话历史连贯
        返回: (是否连贯, 反馈信息)
        """
        if len(self.raw_log) < 2:
            return True, ""  # 对话刚开始，无法检查连贯性
        
        # 获取最近3轮对话作为上下文
        recent_history = self.raw_log[-3:] if len(self.raw_log) >= 3 else self.raw_log
        
        prompt = f"""
        Check if the user's response is coherent with the conversation history.
        
        **Recent Conversation History:**
        {json.dumps(recent_history, ensure_ascii=False)}
        
        **User's Current Response:**
        "{user_response}"
        
        **Review Task:**
        Check if the response:
        1. Responds appropriately to the system's last message
        2. Is consistent with previously expressed views
        3. Does not contain obvious logical contradictions
        4. Maintains conversation flow naturally
        
        **Output Format:**
        If coherent, output: PASS
        If not coherent, output: FAIL|specific reason (e.g., does not respond to system's question, contradicts previous statement, etc.)
        
        **Output:**
        """
        
        try:
            resp = self.client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            result = resp.choices[0].message.content.strip()
            
            if result.upper().startswith("PASS"):
                return True, ""
            elif result.upper().startswith("FAIL"):
                parts = result.split("|", 1)
                reason = parts[1].strip() if len(parts) > 1 else "Response is not coherent with conversation"
                return False, reason
            else:
                return True, ""
        except Exception as e:
            print(f"    [COHERENCE REVIEW ERROR]: {e}")
            return True, ""
    
    def _review_recommendation_quality(self, system_response: str) -> Tuple[bool, str]:
        """
        审查推荐质量：检查系统推荐是否符合用户需求
        返回: (是否符合, 反馈信息)
        """
        if not self.raw_log or len(self.raw_log) == 0:
            return True, ""  # 对话刚开始，无法检查
        
        user_preferences = self.user_profile.get("reflections", {}).get("aesthetic_preferences", [])
        user_last_msg = self.raw_log[-1]["content"] if self.raw_log[-1]["role"] == "user" else ""
        
        # 检查是否包含推荐（有电影标题标记）
        if "**" not in system_response and "recommend" not in system_response.lower():
            return True, ""  # 不是推荐回复，跳过检查
        
        prompt = f"""
        Check if the movie recommendation matches user needs and preferences.
        
        **User Preferences:**
        {json.dumps(user_preferences, ensure_ascii=False)}
        
        **User's Last Message:**
        "{user_last_msg}"
        
        **System's Recommendation:**
        "{system_response}"
        
        **Review Task:**
        Check if the recommendation:
        1. Addresses the user's specific requests
        2. Avoids genres/types the user explicitly dislikes
        3. Provides sufficient justification
        4. Recommends only ONE movie (not multiple)
        5. Is relevant to the conversation context
        
        **Output Format:**
        If the recommendation is good, output: PASS
        If the recommendation has issues, output: FAIL|specific reason (e.g., does not match user preferences, recommends multiple movies, etc.)
        
        **Output:**
        """
        
        try:
            resp = self.client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            result = resp.choices[0].message.content.strip()
            
            if result.upper().startswith("PASS"):
                return True, ""
            elif result.upper().startswith("FAIL"):
                parts = result.split("|", 1)
                reason = parts[1].strip() if len(parts) > 1 else "Recommendation does not match user needs"
                return False, reason
            else:
                return True, ""
        except Exception as e:
            print(f"    [RECOMMENDATION REVIEW ERROR]: {e}")
            return True, ""
    
    def _review_format(self, response: str, role: str) -> Tuple[bool, str]:
        """
        审查回复格式：检查是否符合格式要求
        返回: (是否符合, 反馈信息)
        """
        issues = []
        
        # 检查是否使用了编号列表
        if re.search(r'^\d+[\.\)]\s', response, re.MULTILINE):
            issues.append("Contains numbered list")
        
        # 检查是否使用了项目符号
        if re.search(r'^[-*•]\s', response, re.MULTILINE):
            issues.append("Contains bullet points")
        
        # 对于系统回复，检查是否使用了机器人式标题
        if role == "system":
            if re.search(r'\*\*[^*]+:\*\*', response) or re.search(r'\*\*[^*]+\*\*:', response):
                issues.append("Contains robotic headers like '**Plot:**' or '**Why this fits:**'")
        
        # 检查是否包含多个电影推荐（系统回复）
        if role == "system":
            movie_titles = re.findall(r'\*\*"([^"]+)"\*\*', response)
            if len(movie_titles) > 1:
                issues.append(f"Recommends multiple movies ({len(movie_titles)} movies found)")
        
        if issues:
            return False, f"Format issues: {', '.join(issues)}"
        
        return True, ""
    
    def _review_user_response_comprehensive(self, user_response: str) -> Tuple[bool, str]:
        """
        综合审查用户回复（多维度）
        返回: (是否通过, 反馈信息)
        """
        checks = [
            ("PROFILE", self._review_user_response(user_response)),
            ("COHERENCE", self._review_coherence(user_response)),
        ]
        
        for check_name, (passed, feedback) in checks:
            if not passed:
                return False, f"[{check_name}] {feedback}"
        
        return True, ""
    
    def _review_system_response(self, system_response: str) -> Tuple[bool, str]:
        """
        综合审查系统回复（多维度）
        返回: (是否通过, 反馈信息)
        """
        checks = [
            ("FORMAT", self._review_format(system_response, "system")),
            ("QUALITY", self._review_recommendation_quality(system_response)),
        ]
        
        for check_name, (passed, feedback) in checks:
            if not passed:
                return False, f"[{check_name}] {feedback}"
        
        return True, ""

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
        
        last_msg = init_msg

        while not self.is_finished:
            self.turn_count += 1
            
            print_section(f"ROUND {self.turn_count} START", char="=")

            # --- User Turn ---
            print_section(f"USER TURN (Thinking & Memory Search...)", char="-")

            max_review_retries = 3
            review_retry_count = 0
            user_resp = ""
            review_feedback = ""

            # 生成回复并进行审核，如果不通过则重新生成
            while review_retry_count <= max_review_retries:
                # 生成回复
                user_resp = self.user_agent.reply(last_msg, self.raw_log, self.rejection_count, review_feedback)
                
                # 综合审核回复（多维度）
                print(f"    [REVIEW] Comprehensive review (PROFILE, COHERENCE)...")
                is_compliant, feedback = self._review_user_response_comprehensive(user_resp)
                
                if is_compliant:
                    print(f"    [REVIEW] PASS - All checks passed")
                    break
                else:
                    review_retry_count += 1
                    review_feedback = feedback
                    print(f"    [REVIEW] FAIL - {feedback} (Retry {review_retry_count}/{max_review_retries})")
                    if review_retry_count > max_review_retries:
                        print(f"    [REVIEW] Max retries reached, using current response")
                        break
            
            print_final_response("USER", user_resp)
            self.raw_log.append({"role": "user", "content": user_resp})

            # --- Judge Turn ---
            intent = self._judge_intent(user_resp)
            print(f"    [JUDGE]: {intent} (Rejections: {self.rejection_count})")

            # 状态更新
            if intent == "ACCEPT":
                print("\n *** User Accepted. Conversation End. ***")
                self.is_finished = True
                break
            elif intent == "REJECT":
                self.rejection_count += 1

            if self.turn_count >= config.MAX_TOTAL_TURNS:
                print(" Force stop: Max turns reached.")
                break

            # --- System Turn ---
            print_section(f"SYSTEM TURN (Thinking & Database Search...)", char="-")
            
            # 生成系统回复并进行审核
            max_system_retries = 3
            system_retry_count = 0
            sys_resp = ""
            system_feedback = ""
            
            while system_retry_count <= max_system_retries:
                # 生成回复（传递反馈信息以进行改进）
                sys_resp = self.system_agent.reply(user_resp, self.raw_log, system_feedback)
                
                # 综合审核系统回复（多维度）
                print(f"    [SYSTEM REVIEW] Comprehensive review (FORMAT, QUALITY)...")
                is_compliant, feedback = self._review_system_response(sys_resp)
                
                if is_compliant:
                    print(f"    [SYSTEM REVIEW] PASS - All checks passed")
                    break
                else:
                    system_retry_count += 1
                    system_feedback = feedback
                    print(f"    [SYSTEM REVIEW] FAIL - {feedback} (Retry {system_retry_count}/{max_system_retries})")
                    if system_retry_count > max_system_retries:
                        print(f"    [SYSTEM REVIEW] Max retries reached, using current response")
                        break
            
            print_final_response("SYSTEM", sys_resp)
            self.raw_log.append({"role": "system", "content": sys_resp})
            
            last_msg = sys_resp

        data = {
            "user_id": self.user_profile.get("user_id"),
            "meta_stats": self.user_profile.get("meta_stats"),
            "final_rejection_count": self.rejection_count,
            "turns": self.turn_count,
            "dialogue": self.raw_log
        }
        self.save_results(data)
        return data

    def save_results(self, data: dict):
        if self.enable_result_file and self.output_path:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\nData saved to {self.output_path}")
