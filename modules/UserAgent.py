from config import BaseAgent, MAX_REJECTIONS
import json
from typing import List, Dict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import BaseMessage

class UserAgent(BaseAgent):
    def __init__(self, profile_data: Dict):
        super().__init__(temperature=0.7) 
        self.profile = profile_data
        self.user_id = profile_data.get("user_id", "unknown")
        self.tools = [self._create_memory_tool()]
        self.agent_executor = self._build_agent()

    def _create_memory_tool(self):
        def lookup_memory(query: str) -> str:
            """
            Search past movie memories. 
            Use this to find evidence to SUPPORT your acceptance or rejection.
            """
            hits = []
            memories = self.profile.get("key_memories", [])
            query = query.lower()
            for mem in memories:
                content = f"{mem['movie_title']} {mem['memory_text']}".lower()
                if query in content:
                    hits.append(f"Film: {mem['movie_title']} | Rating: {mem['rating']} | Review: {mem['memory_text']}")
            
            if not hits:
                return "No exact match. Rely on general aesthetic preferences."
            return "\n".join(hits[:3])

        return StructuredTool.from_function(
            func=lookup_memory,
            name="lookup_memory",
            description="Search your movie history. MANDATORY usage before making a decision."
        )

    def _build_agent(self):
        style = self.profile.get("dialogue_style", {})
        reflections = self.profile.get("reflections", {})
        
        system_prompt = f"""
        You are a simulated movie user.
        
        **Profile:**
        - Persona: {reflections.get('spectator_persona')}
        - Tone: {style.get('tone')}
        - General Tastes: {json.dumps(reflections.get('aesthetic_preferences'))}
        
        **CORE INSTRUCTION:**
        1. You are chatting with a Recommender System.
        2. **ALWAYS** use `lookup_memory` to justify your feelings with past experiences.
        3. **ATTITUDE CONTROL:** You will receive a "Current Strategy" in the input. You MUST follow it (e.g., to reject or to accept).
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"), 
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=False)

    def reply(self, message: str, chat_history: List[BaseMessage], rejection_count: int) -> str:
        """
        message: System 的话
        rejection_count: 当前已经拒绝了多少次
        """
        
        if rejection_count < MAX_REJECTIONS:
            strategy_instruction = """
            [CURRENT STRATEGY: BE CRITICAL]
            - If the System is just greeting, chat back casually and hint at your mood.
            - If the System RECOMMENDS a movie, finding a reason to **REJECT** it (unless it is absolutely perfect).
            - Use `lookup_memory` to find a negative example: "I don't want X because I hated [Movie Y]."
            - Reveal your preferences gradually.
            """
        else:
            # 阶段 2: 接受模式 (第三次必须接受)
            strategy_instruction = """
            [CURRENT STRATEGY: ACCEPTANCE]
            - The System has tried hard. You MUST **ACCEPT** this recommendation, even if it's not perfect.
            - Use `lookup_memory` to find a POSITIVE aspect to verify it.
            - Phrase: "You know what, that actually sounds good because..."
            - Thank the system and end the conversation politely.
            """

        full_input = f"System Message: \"{message}\"\n\n{strategy_instruction}"
        
        response = self.agent_executor.invoke({
            "input": full_input,
            "chat_history": chat_history
        })
        return response["output"]


if __name__ == "__main__":
    pass
