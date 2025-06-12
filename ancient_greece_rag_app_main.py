import streamlit as st
import os
import re
import json
import time
import hashlib
import asyncio
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import threading

# LangChain imports
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from datasets import Dataset

# Try to import Ollama, if not available use a mock
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    class ChatOllama:
        def __init__(self, **kwargs):
            self.model = kwargs.get('model', 'llama3.2')
        def invoke(self, prompt):
            return "Ollama not available. Please install langchain-ollama or use alternative LLM."

# Try to import Milvus, if not available use simple vector store
try:
    from langchain_milvus import Milvus
    from pymilvus import connections, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="🏛️ Ancient Greece RAG System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #1e40af, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card {
    background: #f8fafc;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #3b82f6;
    margin: 0.5rem 0;
}
.document-card {
    background: #f1f5f9;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    border: 1px solid #e2e8f0;
}
.success-box {
    background: #dcfce7;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #22c55e;
    margin: 1rem 0;
}
.info-box {
    background: #dbeafe;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #3b82f6;
    margin: 1rem 0;
}
.warning-box {
    background: #fef3c7;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #f59e0b;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Set event loop policy for Windows
if os.name == 'nt':  # Windows
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Ensure event loop exists
def ensure_event_loop():
    """Ensure there's an event loop in the current thread"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

# Configuration Class
class Config:
    def __init__(self):
        # Default paths - users should update these
        self.DATA_DIR = './ancient_greece_data'  # Default local path
        self.MILVUS_HOST = "127.0.0.1"
        self.MILVUS_PORT = "19530"
        self.COLLECTION_NAME = "AncientGreece_DocLevel"
        self.EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
        self.RETRIEVAL_K = 3
        self.LLM_MODEL = "llama3.2"
        self.LLM_BASE_URL = "http://localhost:11434"

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = Config()
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'eval_results' not in st.session_state:
    st.session_state.eval_results = None
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

# Document Loading Functions
def load_documents(data_dir):
    """Load documents without chunking - each file as one document"""
    try:
        if not os.path.exists(data_dir):
            return None, f"Directory not found: {data_dir}"
        
        # Get all .txt files
        txt_files = list(Path(data_dir).glob('**/*.txt'))
        if not txt_files:
            return None, f"No .txt files found in: {data_dir}"
        
        docs = []
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Basic cleaning
                content = re.sub(r'\s+', ' ', content).strip()
                
                doc = Document(
                    page_content=content,
                    metadata={'source': str(file_path)}
                )
                docs.append(doc)
            except Exception as e:
                st.warning(f"Could not load {file_path}: {str(e)}")
        
        return docs, f"Successfully loaded {len(docs)} documents"
    except Exception as e:
        return None, f"Error loading documents: {str(e)}"

# Simple Vector Store (fallback when Milvus is not available)
class SimpleVectorStore:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings
        self.vectors = None
        self._create_vectors()
    
    def _create_vectors(self):
        """Create vector representations of documents"""
        try:
            texts = [doc.page_content for doc in self.documents]
            # Use embedding model to create vectors
            self.vectors = self.embeddings.embed_documents(texts)
        except Exception as e:
            st.error(f"Error creating vectors: {str(e)}")
            # Fallback to TF-IDF
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.vectors = vectorizer.fit_transform(texts).toarray()
    
    def similarity_search(self, query, k=3):
        """Simple similarity search"""
        try:
            if isinstance(self.vectors, list):
                # Using embedding vectors
                query_vector = self.embeddings.embed_query(query)
                similarities = [
                    np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
                    for doc_vector in self.vectors
                ]
            else:
                # Using TF-IDF vectors
                query_vector = self.embeddings.embed_query(query)
                similarities = cosine_similarity([query_vector], self.vectors)[0]
            
            # Get top k documents
            top_indices = np.argsort(similarities)[::-1][:k]
            return [self.documents[i] for i in top_indices]
        except Exception as e:
            st.error(f"Error in similarity search: {str(e)}")
            return self.documents[:k]  # Return first k documents as fallback

# Milvus Vector Store Manager
class MilvusVectorStore:
    def __init__(self):
        self.metadata_file = Path("./simple_vector_cache/metadata.json")
        Path("./simple_vector_cache").mkdir(exist_ok=True)
    
    def get_content_hash(self, documents):
        """Generate hash to check if documents changed"""
        content = "".join([f"{doc.metadata.get('source', '')}{doc.page_content}" for doc in documents])
        return hashlib.md5(content.encode()).hexdigest()
    
    def load_or_create(self, documents, embedding, config):
        """Load existing vector store or create new one"""
        try:
            current_hash = self.get_content_hash(documents)
            
            # Check if we can reuse existing vector store
            if self.metadata_file.exists():
                try:
                    with open(self.metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    if (metadata.get('hash') == current_hash and 
                        metadata.get('collection') == config.COLLECTION_NAME):
                        
                        # Connect and return existing
                        connections.connect(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
                        
                        return Milvus(
                            embedding_function=embedding,
                            connection_args={"host": config.MILVUS_HOST, "port": config.MILVUS_PORT},
                            collection_name=config.COLLECTION_NAME
                        ), "Using existing Milvus vector store"
                except:
                    pass
            
            # Create new vector store
            connections.connect(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
            
            vectorstore = Milvus.from_documents(
                documents=documents,
                embedding=embedding,
                connection_args={"host": config.MILVUS_HOST, "port": config.MILVUS_PORT},
                collection_name=config.COLLECTION_NAME,
                drop_old=True
            )
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump({
                    'hash': current_hash,
                    'collection': config.COLLECTION_NAME,
                    'doc_count': len(documents),
                    'created': datetime.now().isoformat()
                }, f)
            
            return vectorstore, "Created new Milvus vector store"
        
        except Exception as e:
            st.warning(f"Milvus not available: {str(e)}. Using simple vector store.")
            return SimpleVectorStore(documents, embedding), "Using simple vector store (fallback)"

# Hybrid Retriever
class SimpleHybridRetriever:
    def __init__(self, vectorstore, documents, k=3):
        self.vectorstore = vectorstore
        self.documents = documents
        self.k = k
        self.setup_tfidf()
    
    def setup_tfidf(self):
        """Setup TF-IDF for keyword search"""
        try:
            texts = [doc.page_content for doc in self.documents]
            self.tfidf = TfidfVectorizer(
                max_features=3000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.9
            )
            self.tfidf_matrix = self.tfidf.fit_transform(texts)
        except Exception as e:
            st.error(f"Error setting up TF-IDF: {str(e)}")
            # Create a simple fallback
            self.tfidf = None
            self.tfidf_matrix = None
    
    def semantic_search(self, query):
        """Semantic search using embeddings"""
        try:
            return self.vectorstore.similarity_search(query, k=self.k)
        except Exception as e:
            st.error(f"Error in semantic search: {str(e)}")
            return self.documents[:self.k]  # Fallback
    
    def keyword_search(self, query):
        """Keyword search using TF-IDF"""
        try:
            if self.tfidf is None or self.tfidf_matrix is None:
                return []
            
            query_vec = self.tfidf.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top documents
            top_indices = np.argsort(similarities)[::-1][:self.k]
            return [self.documents[idx] for idx in top_indices if similarities[idx] > 0]
        except Exception as e:
            st.error(f"Error in keyword search: {str(e)}")
            return []
    
    def hybrid_search(self, query):
        """Combine semantic and keyword search"""
        try:
            semantic_docs = self.semantic_search(query)
            keyword_docs = self.keyword_search(query)
            
            # Simple combination: take best from each
            seen_sources = set()
            combined_docs = []
            
            # Add semantic results first
            for doc in semantic_docs:
                source = doc.metadata.get('source', '')
                if source not in seen_sources:
                    combined_docs.append(doc)
                    seen_sources.add(source)
            
            # Add keyword results if we need more
            for doc in keyword_docs:
                if len(combined_docs) >= self.k:
                    break
                source = doc.metadata.get('source', '')
                if source not in seen_sources:
                    combined_docs.append(doc)
                    seen_sources.add(source)
            
            return combined_docs[:self.k]
        except Exception as e:
            st.error(f"Error in hybrid search: {str(e)}")
            return self.documents[:self.k]  # Fallback

# Evaluation Functions
def text_similarity(text1, text2):
    """Calculate text similarity using word overlap"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    return len(words1.intersection(words2)) / len(words1.union(words2))

def evaluate_system(dataset):
    """Evaluate the RAG system with multiple metrics"""
    
    def context_precision(dataset):
        total_precision = 0
        for item in dataset:
            contexts = item['contexts']
            ground_truth = item['ground_truth']
            
            relevant_count = 0
            for context in contexts:
                if text_similarity(context, ground_truth) > 0.2:
                    relevant_count += 1
            
            precision = relevant_count / len(contexts) if contexts else 0
            total_precision += precision
        
        return total_precision / len(dataset)
    
    def context_recall(dataset):
        total_recall = 0
        for item in dataset:
            contexts = item['contexts']
            ground_truth = item['ground_truth']
            
            gt_words = set(ground_truth.lower().split())
            context_words = set(" ".join(contexts).lower().split())
            
            if gt_words:
                covered = len(gt_words.intersection(context_words))
                recall = covered / len(gt_words)
                total_recall += recall
        
        return total_recall / len(dataset)
    
    def answer_relevancy(dataset):
        total_relevancy = 0
        for item in dataset:
            question = item['question']
            answer = item['answer']
            
            q_words = set(question.lower().split())
            a_words = set(answer.lower().split())
            
            if q_words:
                overlap = len(q_words.intersection(a_words))
                relevancy = overlap / len(q_words)
                total_relevancy += relevancy
        
        return total_relevancy / len(dataset)
    
    def faithfulness(dataset):
        total_faithfulness = 0
        for item in dataset:
            answer = item['answer']
            contexts = item['contexts']
            
            answer_words = set(answer.lower().split())
            context_words = set(" ".join(contexts).lower().split())
            
            if answer_words:
                supported = len(answer_words.intersection(context_words))
                faithfulness_score = supported / len(answer_words)
                total_faithfulness += faithfulness_score
        
        return total_faithfulness / len(dataset)
    
    cp = context_precision(dataset)
    cr = context_recall(dataset)
    ar = answer_relevancy(dataset)
    f = faithfulness(dataset)
    
    return {
        'context_precision': cp,
        'context_recall': cr,
        'answer_relevancy': ar,
        'faithfulness': f,
        'overall_score': (cp + cr + ar + f) / 4
    }

# Main App Layout
def main():
    st.markdown('<h1 class="main-header">🏛️ Ancient Greece RAG System</h1>', unsafe_allow_html=True)
    
    # Check system dependencies
    show_dependency_warnings()
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Data Directory
        data_dir = st.text_input(
            "📁 Data Directory", 
            value=st.session_state.config.DATA_DIR,
            help="Path to directory containing Ancient Greece text files"
        )
        st.session_state.config.DATA_DIR = data_dir
        
        # Model Settings
        st.subheader("🤖 Model Settings")
        st.session_state.config.EMBEDDING_MODEL = st.selectbox(
            "Embedding Model",
            ["BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"],
            index=0
        )
        
        st.session_state.config.LLM_MODEL = st.text_input(
            "LLM Model", 
            value=st.session_state.config.LLM_MODEL
        )
        
        st.session_state.config.LLM_BASE_URL = st.text_input(
            "LLM Base URL", 
            value=st.session_state.config.LLM_BASE_URL
        )
        
        # Retrieval Settings
        st.subheader("🔍 Retrieval Settings")
        st.session_state.config.RETRIEVAL_K = st.slider(
            "Number of Documents to Retrieve",
            min_value=1,
            max_value=10,
            value=st.session_state.config.RETRIEVAL_K
        )
        
        # Milvus Settings (only show if available)
        if MILVUS_AVAILABLE:
            st.subheader("💾 Vector Database")
            st.session_state.config.MILVUS_HOST = st.text_input(
                "Milvus Host", 
                value=st.session_state.config.MILVUS_HOST
            )
            st.session_state.config.MILVUS_PORT = st.text_input(
                "Milvus Port", 
                value=st.session_state.config.MILVUS_PORT
            )
        
        # Initialize System Button
        if st.button("🚀 Initialize System", type="primary"):
            initialize_system()
        
        # Reset System Button
        if st.session_state.system_initialized:
            if st.button("🔄 Reset System", type="secondary"):
                reset_system()
    
    # Main Content Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 System Status", 
        "❓ Ask Questions", 
        "📄 Document Explorer", 
        "📊 Evaluation", 
        "📈 Analytics"
    ])
    
    with tab1:
        show_system_status()
    
    with tab2:
        show_question_interface()
    
    with tab3:
        show_document_explorer()
    
    with tab4:
        show_evaluation_interface()
    
    with tab5:
        show_analytics()

def show_dependency_warnings():
    """Show warnings about missing dependencies"""
    warnings = []
    
    if not MILVUS_AVAILABLE:
        warnings.append("🟡 Milvus not available - using simple vector store fallback")
    
    if not OLLAMA_AVAILABLE:
        warnings.append("🟡 Ollama not available - LLM functionality limited")
    
    if warnings:
        with st.expander("⚠️ System Warnings", expanded=False):
            for warning in warnings:
                st.markdown(f'<div class="warning-box">{warning}</div>', unsafe_allow_html=True)

def reset_system():
    """Reset the system state"""
    st.session_state.documents = None
    st.session_state.vectorstore = None
    st.session_state.retriever = None
    st.session_state.llm = None
    st.session_state.system_initialized = False
    st.session_state.qa_history = []
    st.session_state.eval_results = None
    st.success("✅ System reset successfully!")

def initialize_system():
    """Initialize the RAG system components"""
    try:
        # Ensure event loop
        ensure_event_loop()
        
        with st.spinner("🔄 Initializing RAG System..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load documents
            status_text.text("📚 Loading documents...")
            progress_bar.progress(20)
            
            docs, message = load_documents(st.session_state.config.DATA_DIR)
            if docs is None:
                st.error(message)
                return
            
            st.session_state.documents = docs
            
            # Setup embeddings
            status_text.text("🧠 Setting up embeddings...")
            progress_bar.progress(40)
            
            try:
                embedding = HuggingFaceEmbeddings(
                    model_name=st.session_state.config.EMBEDDING_MODEL,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
            except Exception as e:
                st.error(f"Error loading embedding model: {str(e)}")
                return
            
            # Setup vector store
            status_text.text("💾 Setting up vector store...")
            progress_bar.progress(60)
            
            try:
                if MILVUS_AVAILABLE:
                    vector_manager = MilvusVectorStore()
                    vectorstore, vs_message = vector_manager.load_or_create(
                        docs, embedding, st.session_state.config
                    )
                else:
                    vectorstore = SimpleVectorStore(docs, embedding)
                    vs_message = "Using simple vector store"
                
                st.session_state.vectorstore = vectorstore
            except Exception as e:
                st.error(f"Error setting up vector store: {str(e)}")
                return
            
            # Setup retriever
            status_text.text("🔍 Setting up retriever...")
            progress_bar.progress(80)
            
            try:
                retriever = SimpleHybridRetriever(
                    vectorstore, docs, k=st.session_state.config.RETRIEVAL_K
                )
                st.session_state.retriever = retriever
            except Exception as e:
                st.error(f"Error setting up retriever: {str(e)}")
                return
            
            # Setup LLM
            status_text.text("🤖 Setting up LLM...")
            progress_bar.progress(90)
            
            try:
                if OLLAMA_AVAILABLE:
                    llm = ChatOllama(
                        model=st.session_state.config.LLM_MODEL,
                        base_url=st.session_state.config.LLM_BASE_URL,
                        temperature=0.1
                    )
                else:
                    llm = ChatOllama()  # Mock LLM
                
                st.session_state.llm = llm
            except Exception as e:
                st.warning(f"LLM setup warning: {str(e)}")
                # Continue with mock LLM
                st.session_state.llm = ChatOllama()
            
            progress_bar.progress(100)
            status_text.text("✅ System initialized successfully!")
            
            st.session_state.system_initialized = True
            st.success(f"✅ System initialized! {message}. {vs_message}.")
            
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        st.error("Please check your configuration and try again.")

def show_system_status():
    """Display system status and information"""
    st.header("📋 System Status")
    
    if not st.session_state.system_initialized:
        st.markdown("""
        <div class="info-box">
        <h3>🚀 Welcome to the Ancient Greece RAG System</h3>
        <p>This system allows you to ask questions about Ancient Greece using a sophisticated 
        Retrieval-Augmented Generation (RAG) approach.</p>
        <p><strong>To get started:</strong></p>
        <ol>
        <li>Configure your settings in the sidebar</li>
        <li>Make sure your data directory contains Ancient Greece text files</li>
        <li>Click "Initialize System" to load documents and setup the vector store</li>
        <li>Start asking questions in the "Ask Questions" tab</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Show example data structure
        st.subheader("📁 Expected Data Structure")
        st.code("""
        ancient_greece_data/
        ├── document1.txt
        ├── document2.txt
        ├── subfolder/
        │   ├── document3.txt
        │   └── document4.txt
        └── ...
        """)
    else:
        # System Information
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📚 Documents Loaded", len(st.session_state.documents) if st.session_state.documents else 0)
        
        with col2:
            st.metric("🔍 Retrieval K", st.session_state.config.RETRIEVAL_K)
        
        with col3:
            st.metric("❓ Questions Asked", len(st.session_state.qa_history))
        
        with col4:
            avg_time = np.mean([qa['time'] for qa in st.session_state.qa_history]) if st.session_state.qa_history else 0
            st.metric("⏱️ Avg Response Time", f"{avg_time:.2f}s")
        
        # System Configuration Display
        st.subheader("⚙️ Current Configuration")
        config_data = {
            "Setting": [
                "Data Directory", "Embedding Model", "LLM Model", "LLM Base URL",
                "Retrieval K", "Vector Store Type"
            ],
            "Value": [
                st.session_state.config.DATA_DIR,
                st.session_state.config.EMBEDDING_MODEL,
                st.session_state.config.LLM_MODEL,
                st.session_state.config.LLM_BASE_URL,
                st.session_state.config.RETRIEVAL_K,
                "Milvus" if MILVUS_AVAILABLE else "Simple"
            ]
        }
        st.dataframe(pd.DataFrame(config_data), use_container_width=True)
        
        # Document Statistics
        if st.session_state.documents:
            st.subheader("📊 Document Statistics")
            
            doc_stats = []
            for doc in st.session_state.documents:
                source = Path(doc.metadata.get('source', 'Unknown')).name
                word_count = len(doc.page_content.split())
                char_count = len(doc.page_content)
                doc_stats.append({
                    'Document': source,
                    'Word Count': word_count,
                    'Character Count': char_count
                })
            
            df_stats = pd.DataFrame(doc_stats)
            st.dataframe(df_stats, use_container_width=True)

def show_question_interface():
    """Question and answer interface"""
    st.header("❓ Ask Questions About Ancient Greece")
    
    if not st.session_state.system_initialized:
        st.warning("⚠️ Please initialize the system first in the System Status tab.")
        return
    
    # Quick Questions
    st.subheader("🚀 Quick Questions")
    quick_questions = [
        "When did Ancient Greece begin?",
        "Who were the Minoans?",
        "What were Greek achievements?",
        "Tell me about ancient Greek architecture",
        "What was the Bronze Age in Greece?"
    ]
    
    selected_quick = st.selectbox("Select a quick question:", [""] + quick_questions)
    
    # Custom Question Input
    st.subheader("💭 Ask Your Own Question")
    question = st.text_area(
        "Enter your question about Ancient Greece:",
        value=selected_quick,
        height=100,
        placeholder="e.g., What were the major achievements of ancient Greek civilization?"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        show_sources = st.checkbox("Show retrieved documents", value=True)
    
    if st.button("🔍 Get Answer", type="primary"):
        if question.strip():
            get_answer(question, show_sources)
        else:
            st.warning("Please enter a question.")
    
    # Question History
    if st.session_state.qa_history:
        st.subheader("📚 Question History")
        
        for i, qa in enumerate(reversed(st.session_state.qa_history[-10:])):  # Show last 10
            with st.expander(f"Q: {qa['question'][:100]}..."):
                st.write(f"**Answer:** {qa['answer']}")
                st.write(f"**Time:** {qa['time']:.2f}s | **Documents:** {qa['doc_count']}")
                if qa.get('sources'):
                    st.write(f"**Sources:** {', '.join(qa['sources'])}")

def get_answer(question, show_sources=True):
    """Get answer for a question using the RAG system"""
    try:
        start_time = time.time()
        
        with st.spinner("🤔 Thinking..."):
            # Check if retriever exists
            if st.session_state.retriever is None:
                st.error("❌ Retriever not initialized. Please reset and reinitialize the system.")
                return
            
            # Retrieve documents
            retrieved_docs = st.session_state.retriever.hybrid_search(question)
            
            if not retrieved_docs:
                st.warning("No relevant documents found.")
                return
            
            # Format context
            formatted_context = []
            sources = []
            for i, doc in enumerate(retrieved_docs, 1):
                source = Path(doc.metadata.get('source', 'Unknown')).name
                sources.append(source)
                content = doc.page_content
                formatted_context.append(f"Document {i} ({source}):\n{content}")
            
            context = "\n\n".join(formatted_context)
            
            # Create prompt
            prompt_template = """You are a historian specializing in Ancient Greece. Answer the question based on the provided documents.

Documents:
{context}

Question: {question}

Answer:"""
            
            # Get answer from LLM
            if st.session_state.llm is None:
                st.error("❌ LLM not initialized. Please reset and reinitialize the system.")
                return
            
            formatted_prompt = prompt_template.format(context=context, question=question)
            
            try:
                response = st.session_state.llm.invoke(formatted_prompt)
                
                if hasattr(response, 'content'):
                    answer = response.content
                else:
                    answer = str(response)
            except Exception as e:
                st.error(f"Error getting LLM response: {str(e)}")
                answer = "Sorry, I couldn't generate an answer due to an LLM error."
            
            elapsed_time = time.time() - start_time
            
            # Display answer
            st.markdown("### 🎯 Answer")
            st.write(answer)
            
            # Display sources if requested
            if show_sources:
                st.markdown("### 📄 Retrieved Documents")
                for i, doc in enumerate(retrieved_docs, 1):
                    source = Path(doc.metadata.get('source', 'Unknown')).name
                    with st.expander(f"Document {i}: {source}"):
                        st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Response Time", f"{elapsed_time:.2f}s")
            with col2:
                st.metric("📄 Documents Retrieved", len(retrieved_docs))
            with col3:
                st.metric("📝 Answer Length", f"{len(answer.split())} words")
            
            # Save to history
            st.session_state.qa_history.append({
                'question': question,
                'answer': answer,
                'time': elapsed_time,
                'doc_count': len(retrieved_docs),
                'sources': sources,
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        st.error(f"❌ Error getting answer: {str(e)}")

def show_document_explorer():
    """Document explorer interface"""
    st.header("📄 Document Explorer")
    
    if not st.session_state.system_initialized or st.session_state.documents is None:
        st.warning("⚠️ Please initialize the system first.")
        return
    
    # Document selection
    doc_names = [Path(doc.metadata.get('source', f'Document {i}')).name 
                for i, doc in enumerate(st.session_state.documents)]
    
    selected_doc = st.selectbox("Select a document to explore:", doc_names)
    
    if selected_doc:
        doc_index = doc_names.index(selected_doc)
        doc = st.session_state.documents[doc_index]
        
        # Document info
        st.subheader(f"📋 Document: {selected_doc}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Word Count", len(doc.page_content.split()))
        with col2:
            st.metric("🔤 Character Count", len(doc.page_content))
        with col3:
            st.metric("📄 Estimated Pages", len(doc.page_content) // 2000 + 1)
        
        # Document content
        st.subheader("📖 Content")
        st.text_area(
            "Document Content:",
            value=doc.page_content,
            height=400,
            disabled=True
        )
        
        # Search within document
        st.subheader("🔍 Search Within Document")
        search_term = st.text_input("Search for specific terms:")
        
        if search_term:
            content_lower = doc.page_content.lower()
            search_lower = search_term.lower()
            
            if search_lower in content_lower:
                # Find all occurrences
                start = 0
                occurrences = []
                while True:
                    pos = content_lower.find(search_lower, start)
                    if pos == -1:
                        break
                    # Get context around the match
                    context_start = max(0, pos - 100)
                    context_end = min(len(doc.page_content), pos + len(search_term) + 100)
                    context = doc.page_content[context_start:context_end]
                    occurrences.append(context)
                    start = pos + 1
                
                st.success(f"Found {len(occurrences)} occurrence(s) of '{search_term}'")
                
                for i, context in enumerate(occurrences, 1):
                    with st.expander(f"Occurrence {i}"):
                        # Highlight the search term
                        highlighted = context.replace(
                            search_term, 
                            f"**{search_term}**"
                        )
                        st.markdown(highlighted)
            else:
                st.info(f"No occurrences of '{search_term}' found.")

def show_evaluation_interface():
    """Evaluation interface"""
    st.header("📊 System Evaluation")
    
    if not st.session_state.system_initialized:
        st.warning("⚠️ Please initialize the system first.")
        return
    
    st.markdown("""
    This section allows you to evaluate the RAG system's performance using predefined questions
    and ground truth answers about Ancient Greece.
    """)
    
    # Evaluation questions
    eval_questions = [
        {
            "question": "When did the story of Ancient Greece begin?",
            "expected_answer": "The story begins around 3000 BCE when a group of seafaring people known as the Minoans settled on the island of Crete.",
            "ground_truth_chunk": "The story begins around 3000 BCE when a group of seafaring people known as the Minoans settled on the island of Crete."
        },
        {
            "question": "Who were the Minoans and where did they settle?",
            "expected_answer": "The Minoans were an advanced society who settled on the island of Crete.",
            "ground_truth_chunk": "The story begins around 3000 BCE when a group of seafaring people known as the Minoans settled on the island of Crete. The Minoans were an advanced society, boasting impressive architecture, intricate artwork, and a thriving trade network."
        },
        {
            "question": "What were some achievements of the Minoans?",
            "expected_answer": "The Minoans were an advanced society, boasting impressive architecture, intricate artwork, and a thriving trade network.",
            "ground_truth_chunk": "The Minoans were an advanced society, boasting impressive architecture, intricate artwork, and a thriving trade network."
        }
    ]
    
    if st.button("🧪 Run Evaluation", type="primary"):
        run_evaluation(eval_questions)
    
    # Display previous evaluation results
    if st.session_state.eval_results:
        display_evaluation_results()

def run_evaluation(eval_questions):
    """Run system evaluation"""
    try:
        with st.spinner("🔄 Running evaluation..."):
            progress_bar = st.progress(0)
            
            dataset_dict = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
            
            for i, item in enumerate(eval_questions):
                progress_bar.progress((i + 1) / len(eval_questions))
                
                question = item["question"]
                ground_truth = item["ground_truth_chunk"]
                
                # Get documents and generate answer
                retrieved_docs = st.session_state.retriever.hybrid_search(question)
                contexts = [doc.page_content for doc in retrieved_docs]
                
                # Format context for LLM
                formatted_context = []
                for j, doc in enumerate(retrieved_docs, 1):
                    source = Path(doc.metadata.get('source', 'Unknown')).name
                    content = doc.page_content
                    formatted_context.append(f"Document {j} ({source}):\n{content}")
                
                context_text = "\n\n".join(formatted_context)
                
                # Create prompt
                prompt_template = """You are a historian specializing in Ancient Greece. Answer the question based on the provided documents.

Documents:
{context}

Question: {question}

Answer:"""
                
                formatted_prompt = prompt_template.format(context=context_text, question=question)
                
                try:
                    response = st.session_state.llm.invoke(formatted_prompt)
                    
                    if hasattr(response, 'content'):
                        answer = response.content
                    else:
                        answer = str(response)
                except Exception as e:
                    answer = f"Error generating answer: {str(e)}"
                
                dataset_dict["question"].append(question)
                dataset_dict["answer"].append(answer)
                dataset_dict["contexts"].append(contexts)
                dataset_dict["ground_truth"].append(ground_truth)
            
            # Create dataset and evaluate
            dataset = Dataset.from_dict(dataset_dict)
            results = evaluate_system(dataset)
            
            st.session_state.eval_results = {
                'results': results,
                'dataset': dataset_dict,
                'timestamp': datetime.now().isoformat()
            }
            
            st.success("✅ Evaluation completed!")
            display_evaluation_results()
            
    except Exception as e:
        st.error(f"❌ Error during evaluation: {str(e)}")

def display_evaluation_results():
    """Display evaluation results"""
    if not st.session_state.eval_results:
        return
    
    results = st.session_state.eval_results['results']
    
    st.subheader("📈 Evaluation Results")
    
    # Metrics overview
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Context Precision", f"{results['context_precision']:.3f}")
    with col2:
        st.metric("Context Recall", f"{results['context_recall']:.3f}")
    with col3:
        st.metric("Answer Relevancy", f"{results['answer_relevancy']:.3f}")
    with col4:
        st.metric("Faithfulness", f"{results['faithfulness']:.3f}")
    with col5:
        st.metric("Overall Score", f"{results['overall_score']:.3f}")
    
    # Performance visualization
    metrics_data = {
        'Metric': ['Context Precision', 'Context Recall', 'Answer Relevancy', 'Faithfulness'],
        'Score': [results['context_precision'], results['context_recall'], 
                 results['answer_relevancy'], results['faithfulness']]
    }
    
    fig = px.bar(
        x=metrics_data['Metric'],
        y=metrics_data['Score'],
        title="RAG System Performance Metrics",
        labels={'x': 'Metrics', 'y': 'Score'},
        color=metrics_data['Score'],
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True, key="evaluation_metrics_bar_chart")
    
    # Performance interpretation
    overall_score = results['overall_score']
    if overall_score >= 0.7:
        st.markdown("""
        <div class="success-box">
        <h3>✅ Excellent Performance!</h3>
        <p>Your RAG system is performing very well with an overall score above 0.7.</p>
        </div>
        """, unsafe_allow_html=True)
    elif overall_score >= 0.5:
        st.markdown("""
        <div class="info-box">
        <h3>👍 Good Performance</h3>
        <p>Your RAG system shows good performance with room for improvement.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("📈 System needs optimization. Consider adjusting retrieval parameters or improving document quality.")
    
    # Detailed results
    with st.expander("📊 Detailed Evaluation Data"):
        dataset = st.session_state.eval_results['dataset']
        eval_df = pd.DataFrame({
            'Question': dataset['question'],
            'Generated Answer': [ans[:100] + "..." if len(ans) > 100 else ans for ans in dataset['answer']],
            'Ground Truth': [gt[:100] + "..." if len(gt) > 100 else gt for gt in dataset['ground_truth']],
            'Context Count': [len(ctx) for ctx in dataset['contexts']]
        })
        st.dataframe(eval_df, use_container_width=True)

def show_analytics():
    """Show system analytics and insights"""
    st.header("📈 System Analytics")
    
    if not st.session_state.qa_history:
        st.info("No questions asked yet. Ask some questions to see analytics!")
        return
    
    # Response time analytics
    times = [qa['time'] for qa in st.session_state.qa_history]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏱️ Response Time Analysis")
        fig_time = px.line(
            x=range(len(times)),
            y=times,
            title="Response Time Over Time",
            labels={'x': 'Question Number', 'y': 'Response Time (seconds)'}
        )
        st.plotly_chart(fig_time, use_container_width=True, key="response_time_line_chart")
    
    with col2:
        st.subheader("📊 Response Time Distribution")
        fig_hist = px.histogram(
            x=times,
            nbins=10,
            title="Response Time Distribution",
            labels={'x': 'Response Time (seconds)', 'y': 'Frequency'}
        )
        st.plotly_chart(fig_hist, use_container_width=True, key="response_time_histogram")
    
    # Document usage analytics
    all_sources = []
    for qa in st.session_state.qa_history:
        if qa.get('sources'):
            all_sources.extend(qa['sources'])
    
    if all_sources:
        source_counts = pd.Series(all_sources).value_counts()
        
        st.subheader("📄 Document Usage Frequency")
        fig_sources = px.bar(
            x=source_counts.index,
            y=source_counts.values,
            title="Most Retrieved Documents",
            labels={'x': 'Document', 'y': 'Retrieval Count'}
        )
        fig_sources.update_xaxes(tickangle=45)
        st.plotly_chart(fig_sources, use_container_width=True, key="document_usage_bar_chart")
    
    # Question length analysis
    question_lengths = [len(qa['question'].split()) for qa in st.session_state.qa_history]
    answer_lengths = [len(qa['answer'].split()) for qa in st.session_state.qa_history]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("❓ Question Length Analysis")
        st.metric("Average Question Length", f"{np.mean(question_lengths):.1f} words")
        st.metric("Shortest Question", f"{min(question_lengths)} words")
        st.metric("Longest Question", f"{max(question_lengths)} words")
    
    with col2:
        st.subheader("📝 Answer Length Analysis")
        st.metric("Average Answer Length", f"{np.mean(answer_lengths):.1f} words")
        st.metric("Shortest Answer", f"{min(answer_lengths)} words")
        st.metric("Longest Answer", f"{max(answer_lengths)} words")
    
    # Performance metrics over time
    if len(st.session_state.qa_history) > 1:
        st.subheader("📈 Performance Trends")
        
        # Moving average of response times
        window_size = min(5, len(times))
        moving_avg = pd.Series(times).rolling(window=window_size).mean()
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=list(range(len(times))),
            y=times,
            mode='markers',
            name='Response Time',
            opacity=0.6
        ))
        fig_trend.add_trace(go.Scatter(
            x=list(range(len(moving_avg))),
            y=moving_avg,
            mode='lines',
            name=f'Moving Average ({window_size})',
            line=dict(color='red', width=2)
        ))
        fig_trend.update_layout(
            title="Response Time Trend",
            xaxis_title="Question Number",
            yaxis_title="Response Time (seconds)"
        )
        st.plotly_chart(fig_trend, use_container_width=True, key="performance_trend_chart")

if __name__ == "__main__":
    main()