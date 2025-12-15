from config import BaseAgent
import json
from typing import List, Dict
import config

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import BaseMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class SystemAgent(BaseAgent):
    def __init__(self):
        super().__init__(temperature=0.4)

        self.vector_db = self._load_local_vector_db() 
        self.tools = [self._create_db_tool()]
        self.agent_executor = self._build_agent()

    def _load_local_vector_db(self):
        """加载本地 Embedding 模型和 FAISS 索引"""
        try:
            print(f"1. Loading Local Embedding Model from: {config.HF_MODEL_LOCAL_PATH} ...")
            
            embeddings = HuggingFaceEmbeddings(
                model_name=config.HF_MODEL_LOCAL_PATH, 
                model_kwargs={'device': config.EMBEDDING_DEVICE}, 
                encode_kwargs={'normalize_embeddings': True} 
            )
            
            print(f"Loading FAISS Index from: {config.FAISS_INDEX_PATH} ...")
            
            vector_db = FAISS.load_local(
                folder_path=config.FAISS_INDEX_PATH, 
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
                index_name="movies"  # 不加后缀
            )
            print("FAISS Database loaded successfully.")
            return vector_db
            
        except Exception as e:
            print(f"Error loading FAISS DB: {e}")
            raise RuntimeError("Failed to load local vector database. Check paths in config.py")

    def _create_db_tool(self):
        def search_movie_database(keywords: str, exclude_titles: str = "") -> str:
            """
            Search the movie library.
            Args:
                keywords: Descriptive search query.
                exclude_titles: Comma-separated list of movie titles to ignore (e.g. movies user already saw or rejected).
            """
            try:
                docs = self.vector_db.similarity_search(keywords, k=10)
                
                if not docs:
                    return "No matching movies found in database."
                
                excludes = [t.strip().lower() for t in exclude_titles.split(",") if t.strip()]
                
                selected_movie = None
                
                for doc in docs:
                    title = doc.metadata.get("title", "Unknown").strip()
                    
                    if title.lower() in excludes:
                        continue
                    
                    selected_movie = doc
                    break 
                
                if not selected_movie:
                    return "Found matches but all were in your exclude list. Try broader keywords."

                meta = selected_movie.metadata
                title = meta.get("title", "Unknown")
                genres = ", ".join(meta.get("genres", []))
                director = ", ".join(meta.get("director", []))
                cast_list = meta.get("cast", [])
                cast = ", ".join(cast_list[:5]) if cast_list else "Unknown"
                overview = meta.get("overview", "No overview.")
                
                return (
                    f"Best Match Found:\n"
                    f"Title: {title}\n"
                    f"Genres: {genres}\n"
                    f"Director: {director}\n"
                    f"Cast: {cast}\n"
                    f"Overview: {overview}\n"
                )
                
            except Exception as e:
                return f"Search Error: {e}"

        return StructuredTool.from_function(
            func=search_movie_database,
            name="search_movie_database",
            description="Semantic search. You MUST provide 'keywords' AND 'exclude_titles' (movies user mentioned/watched) to avoid duplicates."
        )

    def _build_agent(self):
        system_prompt = """
        You are a Movie Recommendation Assistant.
        
        **Strategy:**
        1. **Analyze:** Identify what the user wants AND what they have already mentioned.
        2. **Search:** Use `search_movie_database`.
           - `keywords`: Describe the ideal movie.
           - `exclude_titles`: **CRITICAL**. You MUST list all movie titles the user has mentioned or you have already recommended in this conversation.
        3. **Recommend:** The tool will return the SINGLE BEST match. Use the details (Director, Plot) to sell it to the user.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=False)

    def reply(self, message: str, chat_history: List[BaseMessage]) -> str:
        return self.agent_executor.invoke({"input": message, "chat_history": chat_history})["output"]


if __name__ == "__main__":
    pass
