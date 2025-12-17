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
        memories = self.profile.get("key_memories", [])
        
        self.memory_tool = MemoryRetriever(memories)

        system_message=f"""
        You are a movie enthusiast chatting with an AI.
        
        **YOUR PROFILE:**
        - **Persona:** {reflections.get('spectator_persona')}
        - **Preferences:** {json.dumps(reflections.get('aesthetic_preferences'))}
        
        **SPEAKING STYLE:**
        - **SHORT & SNAPPY:** Keep messages not too long. Like a chat app.
        - **DIRECT:** Don't explain your whole life story. Just react to the recommendation.
        - **FOCUS:** Focus on ONE thing you like or hate at a time.
        
        **TOOL USAGE:**
        - You CAN use `lookup_memory` if you really need to recall a specific movie.
        
        **FINISH:**
        - ALWAYS end with **"TERMINATE"**.
        """

        self.assistant = autogen.AssistantAgent(
            name="User_Simulator",
            system_message=system_message,
            llm_config=config.LLM_CONFIG,
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

        autogen.register_function(
            lookup_memory_wrapper,  
            caller=self.assistant,
            executor=self.executor,
            name="lookup_memory",
            description="Search your movie history for evidence. MANDATORY usage."
        )

    def reply(self, system_msg: str, chat_history: list, rejection_count: int) -> str:

        history_str = ""
        if chat_history:
            for msg in chat_history[-10:]:
                role = "System" if msg['role'] == "system" else "You (User)"
                history_str += f"{role}: {msg['content']}\n"

        if rejection_count < config.MAX_REJECTIONS:
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