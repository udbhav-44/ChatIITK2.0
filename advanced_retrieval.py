from typing import List, Dict
from langchain.retrievers import ParentDocumentRetriever, BM25Retriever
from langchain.retrievers.document_compressors import DocumentCompressor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever, MergerRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np

class AdvancedRetriever:
    def __init__(self, vectorstore: Chroma, embeddings, llm):
        """Initialize advanced retrieval components."""
        # Parent Document Retriever setup
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        
        # Initialize BM25 for hybrid search
        all_docs = vectorstore.get()
        if all_docs["documents"]:
            tokenized_corpus = [doc.split() for doc in all_docs["documents"]]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.corpus = all_docs["documents"]
        else:
            self.bm25 = None
            self.corpus = []
        
        self.parent_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": 6}
        )

        # Embeddings-based reranking
        self.embeddings_filter = EmbeddingsFilter(
            embeddings=embeddings,
            similarity_threshold=0.7
        )

        # Contextual compression
        self.compression_retriever = ContextualCompressionRetriever(
            base_retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
            base_compressor=self.embeddings_filter
        )

    def _expand_query(self, query: str, llm) -> List[str]:
        """Expand the query using LLM."""
        prompt = f"Given the query: '{query}', generate 2 alternative ways to ask the same question. Return only the questions, one per line."
        response = llm.predict(prompt)
        expanded_queries = [query] + [q.strip() for q in response.split('\n') if q.strip()]
        return expanded_queries[:2]  # Limit to original + 1 expansion for efficiency

    async def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using multiple retrieval techniques."""
        # Query expansion
        expanded_queries = self._expand_query(query, self.llm)
        
        # Dense retrieval
        parent_docs = []
        compressed_docs = []
        for q in expanded_queries:
            parent_docs.extend(self.parent_retriever.get_relevant_documents(q))
            compressed_docs.extend(self.compression_retriever.get_relevant_documents(q))
            
        # Sparse retrieval (BM25)
        if self.bm25:
            tokenized_query = query.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_k_indices = np.argsort(bm25_scores)[-4:][::-1]  # Get top 4 docs
            bm25_docs = [Document(page_content=self.corpus[i]) for i in top_k_indices if bm25_scores[i] > 0]
        else:
            bm25_docs = []
        
        # Combine and deduplicate documents
        all_docs = parent_docs + compressed_docs
        seen = set()
        unique_docs = []
        
        for doc in all_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
        
        # Combine all retrieval methods
        all_docs = parent_docs + compressed_docs + bm25_docs
        
        # Deduplicate and rank
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
        
        # Return top 4 most relevant documents
        return unique_docs[:4]
