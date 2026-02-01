import streamlit as st
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="ScholarAgent - Research Hunter",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = None
if 'bookmarks' not in st.session_state:
    st.session_state.bookmarks = {}
if 'search_count' not in st.session_state:
    st.session_state.search_count = 0
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'expanded_query' not in st.session_state:
    st.session_state.expanded_query = {}
if 'original_query' not in st.session_state:
    st.session_state.original_query = ""
if 'interpreted_query' not in st.session_state:
    st.session_state.interpreted_query = ""
if 'query_analyzed' not in st.session_state:
    st.session_state.query_analyzed = False
if 'user_approved' not in st.session_state:
    st.session_state.user_approved = False
if 'user_feedback' not in st.session_state:
    st.session_state.user_feedback = ""
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False

# Main welcome page
st.title("ScholarAgent - Research Hunter")
st.markdown("### Intelligent Research Assistant System")

st.markdown("""
ScholarAgent is an intelligent research assistant system designed for researchers, 
capable of simulating the complete research journey of real researchers:

**Core Features:**

- **Quick Search**: Intelligent AI query understanding, multi-source search, real-time results
- **Paper Library**: Save important papers to your personal library for further analysis
- **Deep Analysis**: Compare, summarize, and generate insights from saved papers

**User Guide:**
1. Select a feature page from the left navigation
2. Configure LLM provider and API key
3. Start your research exploration journey
""")

st.markdown("---")

# Global settings
with st.sidebar:
    st.header("Global Settings")
    
    st.header("LLM Settings")
    llm_provider = st.selectbox(
        "Select LLM Provider",
        options=["openai", "qianwen", "deepseek", "gemini", "openrouter"],
        index=0
    )
    
    st.header("API Keys")
    if llm_provider == "openai":
        api_key = st.text_input("OpenAI API Key", type="password")
    elif llm_provider == "qianwen":
        api_key = st.text_input("Qwen API Key", type="password")
    elif llm_provider == "deepseek":
        api_key = st.text_input("DeepSeek API Key", type="password")
    elif llm_provider == "gemini":
        api_key = st.text_input("Gemini API Key", type="password")
    elif llm_provider == "openrouter":
        api_key = st.text_input("OpenRouter API Key", type="password")
    else:
        api_key = st.text_input("API Key", type="password")
    
    # Store settings in session state
    st.session_state.llm_provider = llm_provider
    st.session_state.api_key = api_key

st.markdown("**ScholarAgent - Making research smarter, making inspiration brighter!**")
