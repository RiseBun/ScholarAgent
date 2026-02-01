import streamlit as st
import sys
import os
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from backend.content_curator import ContentCurator

st.set_page_config(
    page_title="ScholarAgent - 论文收藏库",
    page_icon="📚",
    layout="wide"
)

st.title("📚 论文收藏库")

# Get global settings from session state
llm_provider = st.session_state.get('llm_provider', 'openai')
api_key = st.session_state.get('api_key', '')

# Sidebar options
with st.sidebar:
    st.header("收藏库设置")
    
    st.header("分析选项")
    enable_analysis = st.checkbox("启用AI分析", value=False)
    
    if enable_analysis:
        analysis_type = st.selectbox(
            "分析类型",
            options=["摘要比较", "主题聚类", "研究趋势", "综合分析"],
            index=0
        )
    
    st.header("管理选项")
    if st.button("清空收藏库"):
        if st.session_state.bookmarks:
            st.session_state.bookmarks = {}
            st.success("收藏库已清空")
        else:
            st.warning("收藏库为空")

# Main library interface
if not st.session_state.bookmarks:
    st.warning("收藏库为空。在搜索页面中点击⭐按钮收藏论文。")
else:
    # Display bookmarked papers
    st.subheader(f"收藏的论文 ({len(st.session_state.bookmarks)} 篇)")
    
    # Sort options
    sort_option = st.selectbox(
        "排序方式",
        options=["按添加时间", "按年份 (最新)", "按引用数 (最高)", "按标题 (A-Z)"],
        index=0
    )
    
    # Get and sort papers
    bookmarked_papers = list(st.session_state.bookmarks.values())
    
    if sort_option == "按年份 (最新)":
        bookmarked_papers.sort(key=lambda x: x.get('year', 0), reverse=True)
    elif sort_option == "按引用数 (最高)":
        bookmarked_papers.sort(key=lambda x: x.get('citation_count', 0), reverse=True)
    elif sort_option == "按标题 (A-Z)":
        bookmarked_papers.sort(key=lambda x: x.get('title', '').lower())
    # Default: 按添加时间 (保持字典插入顺序)
    
    # Display papers
    for i, paper in enumerate(bookmarked_papers, 1):
        with st.container(border=True):
            # Title with link
            st.markdown(f"### {i}. [{paper.get('title', 'Untitled')}]({paper.get('url', '#')})")
            
            # Tags
            tags = []
            if paper.get('is_top_venue', False):
                tags.append("🟡 顶级会议")
            if paper.get('has_code', False):
                tags.append("🟢 有代码")
            if paper.get('is_highly_cited', False):
                tags.append("🔴 高引用")
            if paper.get('is_open_access', False):
                tags.append("🔵 开放获取")
            
            if tags:
                st.markdown(f"**标签:** {' '.join(tags)}")
            
            # Abstract (expandable)
            with st.expander("摘要"):
                st.write(paper.get('abstract', 'No abstract available'))
            
            # Authors and venue
            st.markdown(f"**作者:** {', '.join(paper.get('authors', []))}")
            st.markdown(f"**发表场所:** {paper.get('venue', 'Unknown')}")
            st.markdown(f"**年份:** {paper.get('year', 'Unknown')}")
            st.markdown(f"**引用数:** {paper.get('citation_count', 0)}")
            
            # Remove button
            remove_key = f"remove_{i}_{hash(paper.get('title', ''))}"
            if st.button(f"❌ 移除", key=remove_key):
                # Find and remove the paper from bookmarks
                paper_id = None
                for pid, p in st.session_state.bookmarks.items():
                    if p.get('title') == paper.get('title') and p.get('authors') == paper.get('authors'):
                        paper_id = pid
                        break
                
                if paper_id:
                    del st.session_state.bookmarks[paper_id]
                    st.success(f"已移除: {paper.get('title', 'Untitled')}")
                    st.rerun()

# AI Analysis section
if enable_analysis and st.session_state.bookmarks:
    st.markdown("---")
    st.subheader("🤖 AI 分析")
    
    if st.button("生成分析报告"):
        # Create a status container
        analysis_status = st.container()
        analysis_status.info("正在分析收藏的论文...")
        
        try:
            # Get all bookmarked papers
            bookmarked_papers = list(st.session_state.bookmarks.values())
            
            if analysis_type == "摘要比较":
                # Generate comparison of abstracts
                analysis_status.info("正在比较论文摘要...")
                
                # Extract abstracts
                abstracts = [f"Paper {i+1}: {paper.get('title', 'Untitled')}\nAbstract: {paper.get('abstract', 'No abstract')}" 
                            for i, paper in enumerate(bookmarked_papers)]
                
                # Use ContentCurator to analyze
                curator = ContentCurator()
                analysis = curator.analyze_papers(bookmarked_papers, llm_provider, api_key)
                
                # Display analysis
                with st.expander("摘要比较分析"):
                    st.write(analysis.get('summary', '分析失败'))
                    
                    if analysis.get('comparison'):
                        st.subheader("关键差异")
                        for diff in analysis.get('comparison', []):
                            st.markdown(f"- {diff}")
                            
        except Exception as e:
            logger.error(f"分析失败: {e}")
            analysis_status.error(f"分析失败: {str(e)}")
        
        analysis_status.success("分析完成！")

# Quick stats
if st.session_state.bookmarks:
    st.markdown("---")
    st.subheader("📊 收藏库统计")
    
    # Calculate stats
    total_papers = len(st.session_state.bookmarks)
    papers_with_abstracts = sum(1 for p in st.session_state.bookmarks.values() if p.get('abstract'))
    papers_with_code = sum(1 for p in st.session_state.bookmarks.values() if p.get('has_code', False))
    papers_top_venue = sum(1 for p in st.session_state.bookmarks.values() if p.get('is_top_venue', False))
    papers_highly_cited = sum(1 for p in st.session_state.bookmarks.values() if p.get('is_highly_cited', False))
    
    # Display stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总论文数", total_papers)
        st.metric("有摘要", papers_with_abstracts)
    with col2:
        st.metric("有代码", papers_with_code)
        st.metric("顶级会议", papers_top_venue)
    with col3:
        st.metric("高引用", papers_highly_cited)
        
        # Calculate average citation count
        total_citations = sum(p.get('citation_count', 0) for p in st.session_state.bookmarks.values())
        avg_citations = total_citations / total_papers if total_papers > 0 else 0
        st.metric("平均引用", f"{avg_citations:.1f}")

# Future work suggestions
st.markdown("---")
st.subheader("💡 后续功能")
st.markdown("""
- **批量操作**: 选择多篇论文进行批量分析或导出
- **导出功能**: 导出收藏库为CSV、BibTeX或PDF
- **笔记功能**: 为收藏的论文添加个人笔记
- **引用追踪**: 追踪论文的引用网络和相关工作
- **自动更新**: 定期更新论文的引用数和元数据
""")

st.markdown("**提示:** 收藏的论文存储在会话状态中，刷新页面后仍然保留。")
