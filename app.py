import streamlit as st
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="ScholarAgent - 科研猎手",
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
st.title("🔍 ScholarAgent - 科研猎手")
st.markdown("### 智能科研助手系统")

st.markdown("""
ScholarAgent 是一个专为科研人员设计的智能科研助手系统，能够模拟真实科研工作者的完整科研旅程：

**核心功能：**

- **快速搜索**：智能AI查询理解，多数据源搜索，实时结果展示
- **论文收藏**：将重要论文加入个人图书馆，方便后续分析
- **深度分析**：基于收藏论文进行对比、总结和灵感生成

**使用指南：**
1. 在左侧导航栏选择功能页面
2. 配置LLM提供商和API密钥
3. 开始您的科研探索之旅
""")

st.markdown("---")

# Global settings
with st.sidebar:
    st.header("全局设置")
    
    st.header("LLM设置")
    llm_provider = st.selectbox(
        "选择LLM提供商",
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

st.markdown("**ScholarAgent - 让科研更智能，让灵感更闪耀！** ✨")
