import autogen
import json
from modules.tools import MemoryRetriever  
import config

class UserAgent:
    def __init__(self, profile_data: dict):
        self.profile = profile_data
        self.user_id = profile_data.get("user_id", "unknown")
        
        reflections = self.profile.get("reflections", {})
        style = self.profile.get("dialogue_style", {})
        enable_related_memory = bool(getattr(config, "ENABLE_RELATED_USER_MEMORY", True))
        related_users = self.profile.get("related_users", []) if enable_related_memory else []
        
        self.memory_tool = MemoryRetriever(self.user_id, related_users, enable_related_memory)

        memory_scope = (
            "Similar-user expansion is OFF; rely only on your own memories."
            if not enable_related_memory
            else "You may also draw from similar users' memories when they exist."
        )

        system_message=f"""
        You are a movie enthusiast chatting with an AI.
        
        **YOUR PROFILE:**
        - **Persona:** {reflections.get('spectator_persona')}
        - **Tone:** {style.get('tone', '')}
        - **Preferences:** {json.dumps(reflections.get('aesthetic_preferences'))}
        - **Verbosity:** {style.get('verbosity', '')}
        - **Decision Logic:** {reflections.get('decision_logic', '')}
        
        **YOUR ROLE:**
        - You are the **CLIENT/SEEKER**. The System is the **PROVIDER**.
        - **DO NOT** ask the System about its personal life. Focus on what **YOU** want.

        **SPEAKING STYLE:**
        - **SHORT & SNAPPY:** Keep messages not too long. Like a chat app.
        - **DIRECT:** Don't explain your whole life story. Just react to the recommendation.
        - **FOCUS:** Focus on ONE thing you like or hate at a time.
        
        **TOOL USAGE:**
        - You CAN use `lookup_memory` if you really need to recall a specific movie.
        - Memory policy: {memory_scope}
        - Your response should be based on lookup_memory results.
        
        **FINISH:**
        - ALWAYS end with **"TERMINATE"**.
        """

        self.llm_config = config.LLM_CONFIG
        self.llm_config["temperature"] = 0.7

        self.assistant = autogen.AssistantAgent(
            name="User_Simulator",
            system_message=system_message,
            llm_config=self.llm_config,
        )

        self.executor = autogen.UserProxyAgent(
            name="User_Executor",
            human_input_mode="NEVER",
            code_execution_config=False,
            is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
            default_auto_reply="",
        )

        def lookup_memory_wrapper(query: str) -> str:
            return self.memory_tool.lookup(query)

        memory_tool_desc = "Search your movie history for evidence."
        if enable_related_memory:
            memory_tool_desc += " May include similar users' memories."
        else:
            memory_tool_desc += " Similar-user lookup is disabled."
        memory_tool_desc += " MANDATORY usage."

        autogen.register_function(
            lookup_memory_wrapper,  
            caller=self.assistant,
            executor=self.executor,
            name="lookup_memory",
            description=memory_tool_desc
        )

    def reply(self, system_msg: str, chat_history: list, rejection_count: int, review_feedback: str = "") -> str:

        history_str = ""
        if chat_history:
            for msg in chat_history[-10:]:
                role = "System" if msg['role'] == "system" else "You (User)"
                history_str += f"{role}: {msg['content']}\n"
        
        # 三种状态
        if len(chat_history) <= 1:
            strategy_content = """
            - **MODE:** CASUAL / OPEN
            - **ACTION:** The System is greeting you. Respond naturally about your mood or general movie taste.
            - **TONE:** Friendly but hint at your preferences (defined in your profile).
            - **CONSTRAINT:** Do NOT reject anything yet, just start the conversation.
            """
        elif rejection_count < config.MAX_REJECTIONS:
            strategy_content = """
            - **MODE:** CRITICAL / SKEPTICAL
            - **ACTION:** If the System recommends a movie, find a flaw or express hesitation. Use memory to back up your dislike.
            - **NOTE:** If it's just a greeting, be polite but demanding.
            """
        else:
            strategy_content = """
            - **MODE:** ACCEPTING / ENTHUSIASTIC
            - **ACTION:** The recommendation sounds great. Accept it.
            - **NOTE:** Use memory to find a positive connection.
            """

        feedback_section = ""
        if review_feedback:
            feedback_section = f"""
            ===================================================
            [REVIEW FEEDBACK - MUST ADDRESS]
            Your previous response did not match your PROFILE. Please revise your response.
            Issue: {review_feedback}
            
            **REMEMBER YOUR PROFILE:**
            - Persona: {self.profile.get('reflections', {}).get('spectator_persona', '')}
            - Tone: {self.profile.get('dialogue_style', {}).get('tone', '')}
            - You MUST respond according to your PROFILE characteristics.
            ===================================================
            """
            
        # 拼接 Prompt：将策略包装得更像一条系统指令
        # 这里的 system_msg 是来自推荐系统的回复
        full_prompt = f"""
        [CONVERSATION HISTORY]
        {history_str}

        [INCOMING MESSAGE FROM SYSTEM]
        "{system_msg}"

        ===================================================
        [HIDDEN INSTRUCTION]
        {strategy_content}
        ===================================================
        [FEEDBACK SECTION]
        {feedback_section}
        
        Based on the history (don't repeat yourself) and the new message, respond.
        """
        
        self.executor.clear_history()
        self.assistant.clear_history()

        self.executor.initiate_chat(
            self.assistant,
            message=full_prompt,
            max_turns=6
        )

        last_msg = self.executor.last_message(self.assistant)["content"]
        return last_msg.replace("TERMINATE", "").strip()
