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

        sys_msg = f"""
        You are a movie enthusiast looking for a recommendation from an AI Assistant.
        
        **YOUR PROFILE:**
        - **Persona:** {reflections.get('spectator_persona')}
        - **Speaking Style:** {style.get('tone')}
        - **Preferences:** {json.dumps(reflections.get('aesthetic_preferences'))}
        
        **YOUR ROLE:**
        - You are the **CLIENT/SEEKER**. The System is the **PROVIDER**.
        - **DO NOT** ask the System about its personal life. Focus on what **YOU** want.
        
        **CONVERSATION RULES:**
        1. **NO LISTS:** Speak in full, natural paragraphs.
        2. **BE NATURAL:** Describe your needs organically.
        
        **TOOL USAGE PROTOCOL:**
        1. **Greeting:** Chat back, then pivot to your need.
        2. **Recommendation:** **MUST** use `lookup_memory` to verify.
        
        **DYNAMIC STRATEGY PROTOCOL (CRITICAL):**
        In every user message you receive, there will be a section labeled **[HIDDEN INSTRUCTION]**.
        - This instruction overrides your default behavior.
        - You **MUST** execute the specific emotional state (Critical vs Accepting) defined there.
        - Do not output the instruction text itself, just act it out.
        
        **FINISH:**
        - ALWAYS end your turn with **"TERMINATE"**.
        """

        self.assistant = autogen.AssistantAgent(
            name="User_Simulator",
            system_message=sys_msg,
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
        [INCOMING MESSAGE FROM SYSTEM]
        "{system_msg}"

        ===================================================
        [HIDDEN INSTRUCTION]
        {strategy_content}
        ===================================================
        
        Based on the [INCOMING MESSAGE] and adhering to the [HIDDEN INSTRUCTION], respond to the System.
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