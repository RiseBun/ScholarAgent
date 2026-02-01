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
    page_title="ScholarAgent - Paper Library",
    page_icon="📚",
    layout="wide"
)

st.title("Paper Library")

# Get global settings from session state
llm_provider = st.session_state.get('llm_provider', 'openai')
api_key = st.session_state.get('api_key', '')

# Sidebar options
with st.sidebar:
    st.header("Library Settings")
    
    st.header("Analysis Options")
    enable_analysis = st.checkbox("Enable AI Analysis", value=False)
    
    if enable_analysis:
        analysis_type = st.selectbox(
            "Analysis Type",
            options=["Abstract Comparison", "Topic Clustering", "Research Trends", "Comprehensive Analysis"],
            index=0
        )
    
    st.header("Management Options")
    if st.button("Clear Library"):
        if st.session_state.bookmarks:
            st.session_state.bookmarks = {}
            st.success("Library cleared")
        else:
            st.warning("Library is empty")

# Main library interface
if not st.session_state.bookmarks:
    st.warning("Library is empty. Click the Save button in the Search page to save papers.")
else:
    # Display bookmarked papers
    st.subheader(f"Saved Papers ({len(st.session_state.bookmarks)} papers)")
    
    # Sort options
    sort_option = st.selectbox(
        "Sort By",
        options=["Added Time", "Year (Newest)", "Citations (Highest)", "Title (A-Z)"],
        index=0
    )
    
    # Get and sort papers
    bookmarked_papers = list(st.session_state.bookmarks.values())
    
    if sort_option == "Year (Newest)":
        bookmarked_papers.sort(key=lambda x: x.get('year', 0), reverse=True)
    elif sort_option == "Citations (Highest)":
        bookmarked_papers.sort(key=lambda x: x.get('citation_count', 0), reverse=True)
    elif sort_option == "Title (A-Z)":
        bookmarked_papers.sort(key=lambda x: x.get('title', '').lower())
    # Default: Added Time (keep dictionary insertion order)
    
    # Display papers
    for i, paper in enumerate(bookmarked_papers, 1):
        with st.container(border=True):
            # Title with link
            st.markdown(f"### {i}. [{paper.get('title', 'Untitled')}]({paper.get('url', '#')})")
            
            # Tags
            tags = []
            if paper.get('is_top_venue', False):
                tags.append("Top Venue")
            if paper.get('has_code', False):
                tags.append("Has Code")
            if paper.get('is_highly_cited', False):
                tags.append("Highly Cited")
            if paper.get('is_open_access', False):
                tags.append("Open Access")
            
            if tags:
                st.markdown(f"**Tags:** {' | '.join(tags)}")
            
            # Abstract (expandable)
            with st.expander("Abstract"):
                st.write(paper.get('abstract', 'No abstract available'))
            
            # Authors and venue
            st.markdown(f"**Authors:** {', '.join(paper.get('authors', []))}")
            st.markdown(f"**Venue:** {paper.get('venue', 'Unknown')}")
            st.markdown(f"**Year:** {paper.get('year', 'Unknown')}")
            st.markdown(f"**Citations:** {paper.get('citation_count', 0)}")
            
            # Remove button
            remove_key = f"remove_{i}_{hash(paper.get('title', ''))}"
            if st.button(f"Remove", key=remove_key):
                # Find and remove the paper from bookmarks
                paper_id = None
                for pid, p in st.session_state.bookmarks.items():
                    if p.get('title') == paper.get('title') and p.get('authors') == paper.get('authors'):
                        paper_id = pid
                        break
                
                if paper_id:
                    del st.session_state.bookmarks[paper_id]
                    st.success(f"Removed: {paper.get('title', 'Untitled')}")
                    st.rerun()

# AI Analysis section
if enable_analysis and st.session_state.bookmarks:
    st.markdown("---")
    st.subheader("AI Analysis")
    
    if st.button("Generate Analysis Report"):
        # Create a status container
        analysis_status = st.container()
        analysis_status.info("Analyzing saved papers...")
        
        try:
            # Get all bookmarked papers
            bookmarked_papers = list(st.session_state.bookmarks.values())
            
            if analysis_type == "Abstract Comparison":
                # Generate comparison of abstracts
                analysis_status.info("Comparing paper abstracts...")
                
                # Extract abstracts
                abstracts = [f"Paper {i+1}: {paper.get('title', 'Untitled')}\nAbstract: {paper.get('abstract', 'No abstract')}" 
                            for i, paper in enumerate(bookmarked_papers)]
                
                # Use ContentCurator to analyze
                curator = ContentCurator()
                analysis = curator.analyze_papers(bookmarked_papers, llm_provider, api_key)
                
                # Display analysis
                with st.expander("Abstract Comparison Analysis"):
                    st.write(analysis.get('summary', 'Analysis failed'))
                    
                    if analysis.get('comparison'):
                        st.subheader("Key Differences")
                        for diff in analysis.get('comparison', []):
                            st.markdown(f"- {diff}")
                            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            analysis_status.error(f"Analysis failed: {str(e)}")
        
        analysis_status.success("Analysis complete!")

# Quick stats
if st.session_state.bookmarks:
    st.markdown("---")
    st.subheader("Library Statistics")
    
    # Calculate stats
    total_papers = len(st.session_state.bookmarks)
    papers_with_abstracts = sum(1 for p in st.session_state.bookmarks.values() if p.get('abstract'))
    papers_with_code = sum(1 for p in st.session_state.bookmarks.values() if p.get('has_code', False))
    papers_top_venue = sum(1 for p in st.session_state.bookmarks.values() if p.get('is_top_venue', False))
    papers_highly_cited = sum(1 for p in st.session_state.bookmarks.values() if p.get('is_highly_cited', False))
    
    # Display stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Papers", total_papers)
        st.metric("With Abstract", papers_with_abstracts)
    with col2:
        st.metric("With Code", papers_with_code)
        st.metric("Top Venue", papers_top_venue)
    with col3:
        st.metric("Highly Cited", papers_highly_cited)
        
        # Calculate average citation count
        total_citations = sum(p.get('citation_count', 0) for p in st.session_state.bookmarks.values())
        avg_citations = total_citations / total_papers if total_papers > 0 else 0
        st.metric("Avg Citations", f"{avg_citations:.1f}")

# Future work suggestions
st.markdown("---")
st.subheader("Future Features")
st.markdown("""
- **Batch Operations**: Select multiple papers for batch analysis or export
- **Export Function**: Export library to CSV, BibTeX or PDF
- **Note Taking**: Add personal notes to saved papers
- **Citation Tracking**: Track citation network and related work
- **Auto Update**: Periodically update citation counts and metadata
""")

st.markdown("**Tip:** Saved papers are stored in session state and will persist after page refresh.")
