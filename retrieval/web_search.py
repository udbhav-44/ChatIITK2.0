from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from duckduckgo_search import DDGS
from pydantic import Field, BaseModel

class WebSearchRetriever(BaseRetriever, BaseModel):
    num_results: int = Field(default=3)
    ddgs: DDGS = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        try:
            self.ddgs = DDGS()
        except Exception as e:
            print(f"Warning: Failed to initialize web search: {str(e)}")
            self.ddgs = None

    def get_relevant_documents(self, query: str) -> List[Document]:
        if not self.ddgs:
            return []
            
        try:
            results = list(self.ddgs.text(query, max_results=self.num_results))
            docs = [
                Document(
                    page_content=f"{r['title']}\n{r['body']}",
                    metadata={"source": r['link']}
                ) for r in results
            ]
            return docs
        except Exception as e:
            print(f"Web search error: {str(e)}")
            return []
