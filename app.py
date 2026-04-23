import sys

# --- This block prevents the 'ModuleNotFoundError' on your laptop ---
try:
    __import__('pysqlite3')
    import pysqlite3 # We need to make sure it's actually in sys.modules
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, ModuleNotFoundError, KeyError):
    # If we are on Windows/Local, we just skip this part
    pass
# ------------------------------------------------------------------
# ------------------------------------------------------------------


import streamlit as st
import os
import time
from dotenv import load_dotenv

# --- DEBUG TOOL (Delete this after fixing) ---
if st.sidebar.button("🔍 Check API Status"):
    try:
        test_llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key)
        response = test_llm.invoke("Hello, are you alive?")
        st.sidebar.success("✅ API is responding!")
    except Exception as e:
        st.sidebar.error(f"❌ API Error: {e}")
# ----------------------------------------------

# 2026 Modular LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 1. SETUP & ENVIRONMENT
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Retail Banking Advisor", page_icon="🏦", layout="wide")

# Custom CSS for a professional HCL-style UI
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; }
    .stChatFloatingInputContainer { bottom: 20px; }
    .sidebar-header { font-size: 20px; font-weight: bold; color: #004a99; }
    </style>
    """, unsafe_allow_html=True)

# 2. KNOWLEDGE BASE LOADING (Module 2, 4: RAG)
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Pointing to the database created by ingest.py
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return vector_db

vector_db = load_db()

# 3. LLM INITIALIZATION (Gemini 3 Flash)
# max_retries helps handle minor network blips automatically
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key, max_retries=3)

prompt_template = """You are a Senior Retail Banking Advisor. 

INSTRUCTIONS:
1. First, check the PROVIDED CONTEXT below to see if the answer is there.
2. If the answer is in the context, use it to provide a detailed response and mention the sources.
3. If the question is a general banking query (like "What is retail banking?" or career advice) and NOT in the context, use your general expertise to provide a helpful, professional answer.
4. Only say "I don't know" if the question is completely unrelated to banking or finance.

Context: {context} 
Question: {question} 
Answer:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# 4. SESSION STATE & CRUD LOGIC
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {} 

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

if "clicked_question" not in st.session_state:
    st.session_state.clicked_question = None

def start_new_chat():
    new_id = str(int(time.time()))
    st.session_state.all_chats[new_id] = {"title": "New Conversation", "messages": []}
    st.session_state.active_chat_id = new_id

def delete_chat(chat_id):
    del st.session_state.all_chats[chat_id]
    if st.session_state.active_chat_id == chat_id:
        if st.session_state.all_chats:
            st.session_state.active_chat_id = list(st.session_state.all_chats.keys())[0]
        else:
            start_new_chat()

# Ensure at least one chat exists
if not st.session_state.active_chat_id:
    start_new_chat()

# 5. SIDEBAR UI (Navigation & Settings)
with st.sidebar:
    st.markdown('<p class="sidebar-header">🏦 Banking AI Panel</p>', unsafe_allow_html=True)
    
    if st.button("➕ Start New Chat", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()
    
    st.markdown("---")
    
    # PERMANENT SUGGESTIONS
    st.subheader("💡 Suggested Queries")
    if st.button("📈 Check Loan Rates", key="sug_loan"):
        st.session_state.clicked_question = "What are the current interest rates for personal and home loans?"
    if st.button("💳 Compare Accounts", key="sug_compare"):
        st.session_state.clicked_question = "Can you compare the different savings accounts available?"
    
    st.markdown("---")
    st.subheader("📜 Chat History")

    # Render history with Rename/Delete options
    for chat_id in reversed(list(st.session_state.all_chats.keys())):
        chat_data = st.session_state.all_chats[chat_id]
        is_active = (chat_id == st.session_state.active_chat_id)
        
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(f"💬 {chat_data['title']}", key=f"btn_{chat_id}", 
                         use_container_width=True, 
                         type="primary" if is_active else "secondary"):
                st.session_state.active_chat_id = chat_id
                st.rerun()
        with col2:
            with st.popover("⚙️"):
                new_name = st.text_input("Rename", value=chat_data['title'], key=f"ren_{chat_id}")
                if st.button("Save", key=f"save_{chat_id}"):
                    st.session_state.all_chats[chat_id]['title'] = new_name
                    st.rerun()
                st.divider()
                if st.button("🗑️ Delete", key=f"del_{chat_id}", type="primary"):
                    delete_chat(chat_id)
                    st.rerun()

# 6. MAIN CHAT INTERFACE
active_id = st.session_state.active_chat_id
active_chat = st.session_state.all_chats[active_id]
st.header(f"📍 {active_chat['title']}")

# Display conversation history
for message in active_chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process Input
chat_input = st.chat_input("Ask me anything about retail banking...")
final_query = chat_input or st.session_state.clicked_question

if final_query:
    # Immediately show user question
    with st.chat_message("user"):
        st.markdown(final_query)
    
    # Auto-update title for first-time questions
    if active_chat["title"] == "New Conversation":
        active_chat["title"] = final_query[:25] + "..."

    with st.chat_message("assistant"):
        with st.spinner("Consulting bank documents..."):
            try:
                # API Call
                res = qa_chain.invoke(final_query)
                answer = res["result"]
                
                # ✅ SUCCESS PATH: Save and Rerun
                active_chat["messages"].append({"role": "user", "content": final_query})
                active_chat["messages"].append({"role": "assistant", "content": answer})
                st.markdown(answer)
                
                with st.expander("📚 Verified Sources"):
                    for doc in res["source_documents"]:
                        fname = os.path.basename(doc.metadata.get('source', 'Bank_PDF'))
                        pg = doc.metadata.get('page', 'N/A')
                        st.write(f"🔹 **{fname}** (Page {pg})")
                
                st.session_state.clicked_question = None
                st.rerun()

            except Exception as e:
                # 🛑 ERROR PATH: No Rerun (so the error stays visible)
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    st.error("🚨 **Quota Limit Reached:** You've hit the Gemini Free Tier limit (20 requests). Please wait 60 seconds and try again.")
                else:
                    st.error(f"⚠️ **Technical Error:** {e}")
                
                # Reset click trigger even on failure
                st.session_state.clicked_question = None