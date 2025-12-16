import autogen
from modules.tools import MovieRetriever
import config
import re

class SystemAgent:
    def __init__(self):
        self.retriever = MovieRetriever()
        self.seen_movies = set()
        self.assistant = autogen.AssistantAgent(
            name="System_Assistant",
            system_message="""
            You are a Professional Movie Recommendation Consultant.
            
            **YOUR GOAL:** Help the user find **ONE** perfect movie to watch right now.
            
            **WORKFLOW (STRICTLY FOLLOW ORDER):**
            
            **PHASE 1: DIAGNOSIS (Natural Conversation)**
            - If the user input is vague or just a greeting:
              -> **ACTION:** Ask **ONE** open-ended clarifying question naturally.
              -> **CONSTRAINT:** DO NOT use numbered lists. Ask like a friend.
              -> **DO NOT SEARCH yet.**
            - If the user provides specific preferences:
              -> **GO TO PHASE 2.**
            
            **PHASE 2: SEARCH**
            - Formulate specific search keywords based on the user's request.
            - Call `search_movie_database`.
            
            **PHASE 3: RECOMMENDATION (Single Shot, Natural Flow)**
            - **CRITICAL:** You must recommend **EXACTLY ONE** movie. Pick the BEST match.
            - **STYLE:** Speak like a passionate film buff, not a search engine.
            - **STRUCTURE:**
              1. **The Hook:** Announce the movie enthusiastically.
              2. **The Pitch:** Describe the plot and style naturally in a paragraph.
              3. **The Connection:** In a new paragraph, explain *why* it fits their specific request (acting, mood, setting) without using a list. Connect it to their memories if mentioned.
            
            **CRITICAL RULES:**
            - **ONE MOVIE ONLY:** Even if you have multiple ideas, pick the tool's result.
            - **NO LISTS:** Do NOT use numbered lists (1. 2. 3.) or bullet points in your recommendation. Write in full, flowing paragraphs.
            - **NO ROBOTIC HEADERS:** Do not use headers like "**Why this fits:**" or "**Plot:**". Just speak naturally.
            - **ALWAYS** end your turn with the word **"TERMINATE"**.
            """,
            llm_config=config.LLM_CONFIG,
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
        
        context_prompt = f"User said: '{last_user_input}'. \nRespond to the user."
        
        self.executor.initiate_chat(
            self.assistant,
            message=context_prompt,
            max_turns=6
        )
        
        last_msg = self.executor.last_message(self.assistant)["content"]
        
        return last_msg.replace("TERMINATE", "").strip()