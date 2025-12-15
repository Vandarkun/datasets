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
        You are a simulated movie user.
        
        **Profile:**
        - Persona: {reflections.get('spectator_persona')}
        - Tone: {style.get('tone')}
        - General Tastes: {json.dumps(reflections.get('aesthetic_preferences'))}
        
        **CORE INSTRUCTION:**
        1. You are chatting with a Recommender System.
        2. **ALWAYS** use `lookup_memory` to justify your feelings with past experiences.
        3. **ATTITUDE CONTROL:** You will be given a "Current Strategy" in the user message. Follow it STRICTLY.
        
        Reply TERMINATE after you have generated your response to the system.
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
            strategy = """
            [CURRENT STRATEGY: BE CRITICAL]
            - If the System is just greeting, chat back casually.
            - If the System RECOMMENDS a movie, finding a reason to **REJECT** it.
            - Use `lookup_memory` to find a negative example.
            """
        else:
            strategy = """
            [CURRENT STRATEGY: ACCEPTANCE]
            - The System has tried hard. You MUST **ACCEPT** this recommendation.
            - Use `lookup_memory` to find a POSITIVE aspect to verify it.
            - Phrase: "You know what, that actually sounds good because..."
            """

        full_prompt = f"System Message: \"{system_msg}\"\n\n{strategy}\nRespond to the system."

        self.executor.clear_history()
        self.assistant.clear_history()

        self.executor.initiate_chat(
            self.assistant,
            message=full_prompt,
            max_turns=6
        )

        last_msg = self.executor.last_message(self.assistant)["content"]
        return last_msg.replace("TERMINATE", "").strip()