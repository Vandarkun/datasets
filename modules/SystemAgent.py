import autogen
from modules.tools import MovieRetriever
import config
import re

class SystemAgent:
    def __init__(self):
        self.retriever = MovieRetriever()
        self.seen_movies = set()
        self.llm_config = config.LLM_CONFIG
        self.llm_config["temperature"] = 0.7
        self.assistant = autogen.AssistantAgent(
            name="System_Assistant",
            system_message="""
            You are a Casual Movie Buff Friend.

            **GOAL:** Recommend the **BEST AVAILABLE** movie from the database in **UNDER 50 WORDS**.

            **CRITICAL CONSTRAINTS:**
            1. **DATABASE IS LIMITED:** You do not have every movie in the world.
            2. **TRUST THE TOOL:** If `search_movie_database` returns a movie, **YOU MUST USE IT**. That IS the best match.
            3. **NO RE-SEARCHING:** Do NOT search again. Work with what you have.
            4. **BE A SALESMAN:** Spin the movie positively even if it's not a 100% match.

            **STYLE RULES (TO BREAK THE "PERFECT" LOOP):**
            1. **FORBIDDEN OPENERS:** You are **STRICTLY FORBIDDEN** from starting your response with single-word exclamations like:
            - "Perfect!"
            - "Great!"
            - "Awesome!"
            - "Sure!"
            - "Excellent!"
            2. **DIRECT START:** Start sentences with the movie title, a verb, or a question.
            - *Bad:* "Perfect! I have a movie..."
            - *Good:* "*The Matrix* is exactly what you need."
            - *Good:* "You asked for action? I've got a classic for you."
            3. **VARIETY:** Do not use the same sentence structure twice in a row.

            **WORKFLOW:**
            1. Search ONCE based on keywords.
            2. Take the result.
            3. Recommend it briefly.

            **TERMINATION:**
            - Always end with **"TERMINATE"**.
            """,
            llm_config=self.llm_config,
        )

        # 执行者 Agent (Tool Executor)
        self.executor = autogen.UserProxyAgent(
            name="System_Executor",
            human_input_mode="NEVER",
            code_execution_config=False,  
            is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
            default_auto_reply="",
        )

        def search_wrapper(keywords: str, exclude_titles: str = "") -> str:
            explicit_excludes = [t.strip() for t in exclude_titles.split(",") if t.strip()]
            combined_excludes = self.seen_movies.union(set(explicit_excludes))
            final_exclude_str = ", ".join(list(combined_excludes))
            result = self.retriever.search(keywords, final_exclude_str)
            match = re.search(r"Title:\s*(.*?)(?:\n|$)", result)
            if match:
                found_title = match.group(1).strip()
                self.seen_movies.add(found_title)
            
            if "Title:" in result:
                return f"[SYSTEM HINT: RECOMMEND THIS MOVIE.]\n\n{result}"

            return result

        autogen.register_function(
            search_wrapper,
            caller=self.assistant,
            executor=self.executor,
            name="search_movie_database",
            description="Search the movie library. Returns the best matching movie details."
        )

    def reply(self, last_user_input: str, chat_history: list = None) -> str:
        """
        发起一次内部对话，获取 System 的回复。
        """
        # 将上一轮输入作为 prompt。没有保存完整对话历史        
        # 限制 max_turns（思考turns）
        # 流程通常是: Assistant(Call Tool) -> Executor(Run Tool) -> Assistant(Final Answer)
        
        # 清空 executor 的历史，重新开始一次“思考-行动-回复”的循环
        self.executor.clear_history() 
        self.assistant.clear_history()
        
        history_str = ""
        if chat_history:
            for msg in chat_history[-10:]:
                role = "User" if msg['role'] == "user" else "You (System)"
                history_str += f"{role}: {msg['content']}\n"

        context_prompt = f"""
        [CONVERSATION HISTORY]
        {history_str}
        
        [CURRENT SITUATION]
        User just said: "{last_user_input}"
        
        Based on the history and the new input, respond to the user.
        """
        
        self.executor.initiate_chat(
            self.assistant,
            message=context_prompt,
            max_turns=6
        )
        
        last_msg = self.executor.last_message(self.assistant)["content"]
        
        return last_msg.replace("TERMINATE", "").strip()