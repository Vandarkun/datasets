import autogen
from modules.tools import MovieRetriever
import config

class SystemAgent:
    def __init__(self):
        self.retriever = MovieRetriever()
        
        # 思考者 Agent
        self.assistant = autogen.AssistantAgent(
            name="System_Assistant",
            system_message="""
            You are a Movie Recommendation Assistant.
            
            **Strategy:**
            1. **Analyze:** Identify what the user wants AND what they have already mentioned.
            2. **Search:** Use `search_movie_database` tool.
               - `keywords`: Describe the ideal movie.
               - `exclude_titles`: **CRITICAL**. You MUST list all movie titles the user has mentioned or you have already recommended.
            3. **Recommend:** The tool will return the SINGLE BEST match. Use the details (Director, Plot) to sell it to the user.
            
            Reply TERMINATE when the task is done or you have made a recommendation.
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
            return self.retriever.search(keywords, exclude_titles)

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
        # AutoGen 的 chat history 是自动管理的，但在这种轮次控制的场景下，
        # 我们通常需要把上下文注入进去。为了简单起见，我们将上一轮输入作为 prompt。
        # 如果需要保留完整历史，可以将 chat_history 转换为 AutoGen 的 messages 格式并注入。
        
        # 为了防止 Agent 无限循环，我们限制 max_turns
        # 流程通常是: Assistant(Call Tool) -> Executor(Run Tool) -> Assistant(Final Answer)
        
        # 我们清空 executor 的历史，重新开始一次“思考-行动-回复”的循环
        self.executor.clear_history() 
        self.assistant.clear_history()
        
        # 如果有历史记录，可以通过 prepend 方式加入（这里简化处理，只传当前输入）
        # 实际生产中建议把 chat_history 拼接到 message 中作为 Context
        
        context_prompt = f"User said: '{last_user_input}'. \nRespond to the user."
        
        self.executor.initiate_chat(
            self.assistant,
            message=context_prompt,
            max_turns=6
        )
        
        last_msg = self.executor.last_message(self.assistant)["content"]
        return last_msg.replace("TERMINATE", "").strip()