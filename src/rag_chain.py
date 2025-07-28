"""RAG chain implementation with LLM integration."""

from typing import Dict, Any, List, Optional, Tuple
import time

from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.llms.base import LLM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

from .config import Config
from .embedding_service import EmbeddingService


class RAGChain:
    """RAG chain for question answering with document retrieval."""
    
    def __init__(self):
        self.config = Config()
        self.embedding_service = EmbeddingService()
        self.llm: Optional[LLM] = None
        self.qa_chain: Optional[RetrievalQA] = None
        self.conversational_chain: Optional[ConversationalRetrievalChain] = None
        self.memory: Optional[ConversationBufferWindowMemory] = None
        
        # Custom prompt template
        self.qa_prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context provided, just say that you don't know, don't try to make up an answer.
Always cite the source document when providing information.

Context:
{context}

Question: {question}

Answer with source citations:"""
        
        self.qa_prompt = PromptTemplate(
            template=self.qa_prompt_template,
            input_variables=["context", "question"]
        )
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize the RAG chain."""
        try:
            # Initialize embedding service
            if not self.embedding_service.initialize():
                return {
                    "success": False,
                    "error": "Failed to initialize embedding service"
                }
            
            # Initialize LLM
            llm_provider = self.config.get_available_llm_provider()
            if not llm_provider:
                return {
                    "success": False,
                    "error": "No LLM API key found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY"
                }
            
            if llm_provider == "openai":
                self.llm = ChatOpenAI(
                    model_name=self.config.OPENAI_MODEL,
                    temperature=self.config.TEMPERATURE,
                    max_tokens=self.config.MAX_TOKENS,
                    openai_api_key=self.config.OPENAI_API_KEY
                )
            elif llm_provider == "anthropic":
                self.llm = ChatAnthropic(
                    model=self.config.ANTHROPIC_MODEL,
                    temperature=self.config.TEMPERATURE,
                    max_tokens=self.config.MAX_TOKENS,
                    anthropic_api_key=self.config.ANTHROPIC_API_KEY
                )
            
            # Initialize memory for conversational chain
            self.memory = ConversationBufferWindowMemory(
                k=5,  # Remember last 5 exchanges
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Create QA chain if vector store is ready
            store_status = self.embedding_service.get_store_status()
            if store_status["initialized"]:
                self._create_qa_chains()
            
            return {
                "success": True,
                "llm_provider": llm_provider,
                "vector_store_initialized": store_status["initialized"],
                "ready_for_qa": self.qa_chain is not None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Initialization failed: {str(e)}"
            }
    
    def _create_qa_chains(self) -> None:
        """Create QA chains using the initialized components."""
        if not self.llm:
            raise ValueError("LLM not initialized")
        
        # Get retriever
        retriever = self.embedding_service.get_retriever(
            search_kwargs={"k": 4}
        )
        
        # Create standard QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True
        )
        
        # Create conversational chain
        self.conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=False
        )
    
    def process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process documents and update the QA chain."""
        # Process documents using embedding service
        result = self.embedding_service.process_and_embed_documents(file_paths)
        
        if result["success"]:
            # Create/update QA chains
            try:
                self._create_qa_chains()
                result["qa_chain_ready"] = True
            except Exception as e:
                result["qa_chain_ready"] = False
                result["qa_chain_error"] = str(e)
        
        return result
    
    def ask_question(
        self, 
        question: str, 
        use_conversation: bool = False
    ) -> Dict[str, Any]:
        """Answer a question using the RAG chain."""
        if not self.qa_chain:
            return {
                "success": False,
                "error": "QA chain not initialized. Process documents first."
            }
        
        start_time = time.time()
        
        try:
            if use_conversation and self.conversational_chain:
                # Use conversational chain for follow-up questions
                result = self.conversational_chain({
                    "question": question
                })
                
                answer = result.get("answer", "")
                source_docs = result.get("source_documents", [])
            else:
                # Use standard QA chain
                result = self.qa_chain({
                    "query": question
                })
                
                answer = result.get("result", "")
                source_docs = result.get("source_documents", [])
            
            # Format source documents
            sources = self._format_sources(source_docs)
            
            response_time = time.time() - start_time
            
            return {
                "success": True,
                "question": question,
                "answer": answer,
                "sources": sources,
                "response_time": round(response_time, 2),
                "source_count": len(source_docs)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Question processing failed: {str(e)}",
                "response_time": time.time() - start_time
            }
    
    def _format_sources(self, source_docs: List[Document]) -> List[Dict[str, Any]]:
        """Format source documents for display."""
        sources = []
        
        for i, doc in enumerate(source_docs):
            source_info = {
                "index": i + 1,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata,
                "filename": doc.metadata.get("filename", "Unknown"),
                "source": doc.metadata.get("source", "Unknown"),
                "chunk_id": doc.metadata.get("chunk_id", i)
            }
            sources.append(source_info)
        
        return sources
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        if not self.memory:
            return []
        
        try:
            history = []
            chat_memory = self.memory.chat_memory.messages
            
            for i in range(0, len(chat_memory), 2):
                if i + 1 < len(chat_memory):
                    human_message = chat_memory[i]
                    ai_message = chat_memory[i + 1]
                    
                    history.append({
                        "question": human_message.content,
                        "answer": ai_message.content
                    })
            
            return history
            
        except Exception as e:
            print(f"Error getting conversation history: {str(e)}")
            return []
    
    def clear_conversation_history(self) -> Dict[str, Any]:
        """Clear conversation history."""
        try:
            if self.memory:
                self.memory.clear()
            
            return {
                "success": True,
                "message": "Conversation history cleared"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to clear history: {str(e)}"
            }
    
    def search_documents(self, query: str, k: int = 4) -> Dict[str, Any]:
        """Search documents without generating an answer."""
        return self.embedding_service.search_documents(query, k, include_scores=True)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        store_status = self.embedding_service.get_store_status()
        
        return {
            "llm_initialized": self.llm is not None,
            "llm_provider": self.config.get_available_llm_provider(),
            "qa_chain_ready": self.qa_chain is not None,
            "conversational_chain_ready": self.conversational_chain is not None,
            "vector_store_status": store_status,
            "api_keys_configured": {
                "openai": bool(self.config.OPENAI_API_KEY),
                "anthropic": bool(self.config.ANTHROPIC_API_KEY)
            },
            "conversation_history_length": len(self.get_conversation_history())
        }