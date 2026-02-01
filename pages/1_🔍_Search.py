import streamlit as st
import sys
import os
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from backend.search_engine import PaperFetcher
from backend.tagger import enrich_and_tag
from backend.query_expander import QueryExpander
from backend.content_curator import ContentCurator

st.set_page_config(
    page_title="ScholarAgent - 快速搜索",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Quick Search")

# Get global settings from session state
llm_provider = st.session_state.get('llm_provider', 'openai')
api_key = st.session_state.get('api_key', '')

# Sidebar filters
with st.sidebar:
    st.header("Search Settings")
    
    st.header("Filters")
    must_have_code = st.checkbox("Must have Code")
    top_venue_only = st.checkbox("Top Venue Only")
    highly_cited_only = st.checkbox("Highly Cited Only")
    
    st.header("Year Range")
    year_start = st.slider("Start Year", min_value=2010, max_value=2026, value=2024)
    year_end = st.slider("End Year", min_value=2010, max_value=2026, value=2026)
    year_range = f"{year_start}-{year_end}"
    
    st.header("Results Settings")
    result_limit = st.slider("Number of Papers", min_value=5, max_value=50, value=20, step=5)
    st.info(f"Will search up to {result_limit} papers from arXiv and OpenAlex")

# Main search interface
st.header("What are you researching?")

# Determine which query to display
display_query = st.session_state.interpreted_query if st.session_state.query_analyzed else ""
query_placeholder = "e.g., Mamba architecture for vision"

user_query = st.text_area(
    "Research Query",
    value=display_query,
    placeholder=query_placeholder,
    height=100,
    key="user_query_input"
)

# Create action buttons
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    analyze_button = st.button("🤖 Interpret Query")
with col2:
    search_button = st.button("🔍 Start Search", disabled=not st.session_state.user_approved)
with col3:
    refresh_button = st.button("🔄 Refresh")

# Interpret query when analyze button is clicked
if analyze_button:
    if not user_query:
        st.error("Please enter a research query")
    else:
        # Always update original query to current input
        st.session_state.original_query = user_query
        
        # Reset states for new analysis
        st.session_state.user_approved = False
        # Don't reset feedback_given here to allow multiple feedback cycles
        
        # Create a status container for analysis
        analysis_status = st.container()
        analysis_status.info("Analyzing query...")
        
        try:
            # Use QueryExpander to expand the query
            expander = QueryExpander()
            # If feedback was given, use it to improve interpretation
            if st.session_state.feedback_given:
                # Create a more effective combined query with feedback
                combined_query = f"Current query: {user_query}\nUser feedback: {st.session_state.user_feedback}\nPlease carefully consider the feedback and provide an improved interpretation that addresses the user's concerns. Focus on making the interpretation more accurate and relevant to what the user actually wants."
                expanded_query = expander.expand_query(combined_query, llm_provider, api_key)
            else:
                expanded_query = expander.expand_query(user_query, llm_provider, api_key)
            
            st.session_state.expanded_query = expanded_query
            st.session_state.interpreted_query = expanded_query.get('search_query', user_query)
            st.session_state.query_analyzed = True
            
            analysis_status.success("Query analysis completed!")
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            analysis_status.error(f"Error analyzing query: {str(e)}")

# Display user feedback window if query has been analyzed
if st.session_state.query_analyzed:
    expanded_query = st.session_state.expanded_query
    
    with st.container(border=True):
        st.subheader("💬 Provide Feedback")
        
        # Show current query and interpreted queries
        st.markdown(f"**Current Query:** {st.session_state.original_query}")
        st.markdown(f"**AI Interpretation:** {st.session_state.interpreted_query}")
        st.markdown(f"**Extracted Venue:** {expanded_query.get('venue_filter', 'None')}")
        st.markdown(f"**Research Domain:** {expanded_query.get('domain', 'General')}")
        st.markdown(f"**Explanation:** {expanded_query.get('explanation', 'No explanation available')}")
        
        # User feedback input with better instructions
        user_feedback = st.text_area(
            "Your Feedback (please be specific about what's wrong or how to improve the interpretation)",
            value=st.session_state.user_feedback,
            height=150,
            key="user_feedback_input",
            placeholder="e.g.,\n- I want to focus on more recent research\n- Please include more relevant technical terms\n- My research domain is computer vision, not natural language processing\n- Please don't include irrelevant venues"
        )
        
        # Feedback action buttons
        feedback_col1, feedback_col2 = st.columns(2)
        with feedback_col1:
            if st.button("🔄 Update Interpretation"):
                if user_feedback:
                    st.session_state.user_feedback = user_feedback
                    st.session_state.feedback_given = True
                    # Trigger re-analysis with feedback
                    st.rerun()
                else:
                    st.error("Please provide feedback before updating")
        
        with feedback_col2:
            if st.button("✅ Approve Interpretation"):
                st.session_state.user_approved = True
                st.success("Interpretation approved! You can now start the search.")
                # Force a rerun to update the button state immediately
                st.rerun()

# Trigger search if search or refresh button is clicked and user has approved
if (search_button or refresh_button) and st.session_state.user_approved:
    if not user_query:
        st.error("请输入研究查询")
    else:
        # Determine if it's a new search or page refresh
        is_new_search = search_button or user_query != st.session_state.last_query
        
        if is_new_search:
            # Reset pagination for new search
            st.session_state.current_page = 1
            st.session_state.search_history = []
            st.session_state.last_query = user_query
        else:
            # Increment page for refresh
            st.session_state.current_page += 1
        
        # Increment search count to force refresh
        st.session_state.search_count += 1
        
        # Create a status container to display real-time updates
        status_container = st.container()
        status_container.info("Starting search process...")
        
        logger.info(f"User initiated search with query: '{user_query}'")
        logger.info(f"LLM Provider: {llm_provider}, API Key: {'Provided' if api_key else 'Not provided'}")
        logger.info(f"Filters: Must have Code={must_have_code}, Top Venue Only={top_venue_only}, Highly Cited Only={highly_cited_only}")
        logger.info(f"Year Range: {year_range}")
        logger.info(f"Result Limit: {result_limit}")
        logger.info(f"Current page: {st.session_state.current_page}")
        
        # Add a results container that will be updated incrementally
        results_container = st.container()
        results_header = results_container.empty()
        paper_containers = []
        
        # Initialize variables for streaming results
        all_papers = []
        displayed_count = 0
        keywords = user_query
        filters = {}
        
        # Get the approved expanded query
        expanded_query = st.session_state.expanded_query
        
        # Calculate offset for pagination
        offset = (st.session_state.current_page - 1) * result_limit
        
        # Use optimized search strategy with expanded query
        status_container.info("Searching papers with optimized strategy...")
        logger.info(f"Searching papers with optimized strategy for query: '{expanded_query.get('search_query', user_query)}'")
        
        try:
            fetcher = PaperFetcher()
            
            # Use the expanded query for search
            search_query = expanded_query.get('search_query', user_query)
            venue_filter = expanded_query.get('venue_filter', '')
            
            # Use the new optimized search method with pagination
            papers = fetcher.search_papers(
                search_query, 
                limit=result_limit, 
                year_range=year_range, 
                venue_filter=venue_filter,
                offset=offset
            )
            logger.info(f"Optimized search returned {len(papers)} papers")
            
            # If no papers found, try with broader search
            if not papers:
                status_container.info("No papers found, trying broader search...")
                # Try without venue filter
                papers = fetcher.search_papers(
                    search_query, 
                    limit=result_limit, 
                    year_range=year_range, 
                    venue_filter='',
                    offset=offset
                )
                logger.info(f"Broader search returned {len(papers)} papers")
            
            # Process papers first
            for paper_data in papers:
                # Create Paper object and add tags
                tagged_papers = enrich_and_tag([paper_data])
                if tagged_papers:
                    paper = tagged_papers[0]
                    
                    # Apply filters
                    if (not must_have_code or paper.has_code) and \
                       (not top_venue_only or paper.is_top_venue) and \
                       (not highly_cited_only or paper.is_highly_cited):
                        all_papers.append(paper)
                        displayed_count += 1
            
            # If still no papers after filtering, try without filters
            if not all_papers and papers:
                status_container.info("No papers passed filters, showing all results...")
                for paper_data in papers:
                    tagged_papers = enrich_and_tag([paper_data])
                    if tagged_papers:
                        all_papers.append(tagged_papers[0])
        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            status_container.error(f"Error searching papers: {str(e)}")
        
        # Save current page to history
        st.session_state.search_history.append({
            'page': st.session_state.current_page,
            'query': user_query,
            'expanded_query': expanded_query,
            'papers': all_papers
        })
        
        # Store results in session state
        st.session_state.current_results = all_papers
        
        # Display AI's interpretation of the query (now above results)
        with st.expander("🤖 Final AI Query Interpretation"):
            st.markdown(f"**Original Query:** {user_query}")
            st.markdown(f"**Expanded Query:** {expanded_query.get('search_query', user_query)}")
            st.markdown(f"**Extracted Venue:** {expanded_query.get('venue_filter', 'None')}")
            st.markdown(f"**Research Domain:** {expanded_query.get('domain', 'General')}")
            st.markdown(f"**Explanation:** {expanded_query.get('explanation', 'No explanation available')}")
        
        # Display pagination info (now above results)
        st.markdown(f"**Current Page:** {st.session_state.current_page}")
        if len(st.session_state.search_history) > 1:
            st.markdown(f"**Available Pages:** 1 - {len(st.session_state.search_history)}")
            
            # Add buttons to navigate to previous pages
            with st.expander("📋 View History Pages"):
                for history in st.session_state.search_history:
                    page_num = history['page']
                    if page_num != st.session_state.current_page:
                        if st.button(f"Go to Page {page_num}"):
                            st.session_state.current_page = page_num
                            # Store the selected page's papers to display
                            for h in st.session_state.search_history:
                                if h['page'] == page_num:
                                    st.session_state.current_results = h['papers']
                                    st.session_state.current_expanded_query = h['expanded_query']
                                    break
                            st.rerun()
        
        # Update results header
        if all_papers:
            results_header.subheader(f"Results (Page {st.session_state.current_page}) - Found {len(all_papers)} papers")
        else:
            results_header.subheader(f"Results (Page {st.session_state.current_page}) - No papers found")
            status_container.warning("No papers found. Showing most relevant results...")
        
        # Display papers now
        for paper in all_papers:
            # Create a new container for this paper
            paper_container = results_container.empty()
            paper_containers.append(paper_container)
            
            # Display the new paper
            with paper_container:
                with st.container(border=True):
                    # Title with link
                    st.markdown(f"### [{paper.title}]({paper.url})")
                    
                    # Tags
                    tags = []
                    if paper.is_top_venue:
                        tags.append("🟡 Top Venue")
                    if paper.has_code:
                        tags.append("🟢 Has Code")
                    if paper.is_highly_cited:
                        tags.append("🔴 Highly Cited")
                    if paper.is_open_access:
                        tags.append("🔵 Open Access")
                    
                    if tags:
                        st.markdown(f"**Tags:** {' '.join(tags)}")
                    
                    # Abstract (expandable)
                    with st.expander("Abstract"):
                        st.write(paper.abstract)
                    
                    # Authors and venue
                    st.markdown(f"**Authors:** {', '.join(paper.authors)}")
                    st.markdown(f"**Venue:** {paper.venue}")
                    st.markdown(f"**Year:** {paper.year}")
                    st.markdown(f"**Citations:** {paper.citation_count}")
                    
                    # Bookmark button
                    paper_id = paper.title  # Use title as unique identifier
                    bookmark_key = f"bookmark_{paper_id}_{st.session_state.search_count}"
                    if st.button(f"⭐ Save", key=bookmark_key):
                        # Add paper to bookmarks
                        st.session_state.bookmarks[paper_id] = {
                            'title': paper.title,
                            'authors': paper.authors,
                            'year': paper.year,
                            'venue': paper.venue,
                            'url': paper.url,
                            'abstract': paper.abstract,
                            'citation_count': paper.citation_count,
                            'has_code': paper.has_code,
                            'is_top_venue': paper.is_top_venue,
                            'is_highly_cited': paper.is_highly_cited,
                            'is_open_access': paper.is_open_access
                        }
                        st.toast(f"Saved: {paper.title}")
        
        status_container.success(f"Search completed successfully! Found {len(all_papers)} papers on Page {st.session_state.current_page}.")
        logger.info(f"Search completed successfully! Found {len(all_papers)} papers on Page {st.session_state.current_page}.")

# Display historical results if available
if 'current_results' in st.session_state and st.session_state.current_results:
    if not (search_button or refresh_button or analyze_button):
        # Display historical results
        expanded_query = st.session_state.get('current_expanded_query', st.session_state.expanded_query)
        all_papers = st.session_state.current_results
        
        # Display AI's interpretation of the query
        with st.expander("🤖 最终AI查询解释"):
            st.markdown(f"**原始查询:** {st.session_state.original_query}")
            st.markdown(f"**扩展查询:** {expanded_query.get('search_query', st.session_state.original_query)}")
            st.markdown(f"**提取的Venue:** {expanded_query.get('venue_filter', 'None')}")
            st.markdown(f"**研究领域:** {expanded_query.get('domain', 'General')}")
            st.markdown(f"**解释:** {expanded_query.get('explanation', 'No explanation available')}")
        
        # Display pagination info
        st.markdown(f"**当前页面:** {st.session_state.current_page}")
        if len(st.session_state.search_history) > 1:
            st.markdown(f"**可用页面:** 1 - {len(st.session_state.search_history)}")
            
            # Add buttons to navigate to previous pages
            with st.expander("📋 查看历史页面"):
                for history in st.session_state.search_history:
                    page_num = history['page']
                    if page_num != st.session_state.current_page:
                        if st.button(f"前往页面 {page_num}"):
                            st.session_state.current_page = page_num
                            # Store the selected page's papers to display
                            for h in st.session_state.search_history:
                                if h['page'] == page_num:
                                    st.session_state.current_results = h['papers']
                                    st.session_state.current_expanded_query = h['expanded_query']
                                    break
                            st.rerun()
        
        # Update results header
        results_container = st.container()
        results_header = results_container.empty()
        if all_papers:
            results_header.subheader(f"结果 (页面 {st.session_state.current_page}) - 找到 {len(all_papers)} 篇论文")
        else:
            results_header.subheader(f"结果 (页面 {st.session_state.current_page}) - 未找到论文")
        
        # Display papers
        for paper in all_papers:
            # Create a new container for this paper
            paper_container = results_container.empty()
            
            # Display the paper
            with paper_container:
                with st.container(border=True):
                    # Title with link
                    st.markdown(f"### [{paper.title}]({paper.url})")
                    
                    # Tags
                    tags = []
                    if paper.is_top_venue:
                        tags.append("🟡 顶级会议")
                    if paper.has_code:
                        tags.append("🟢 有代码")
                    if paper.is_highly_cited:
                        tags.append("🔴 高引用")
                    if paper.is_open_access:
                        tags.append("🔵 开放获取")
                    
                    if tags:
                        st.markdown(f"**标签:** {' '.join(tags)}")
                    
                    # Abstract (expandable)
                    with st.expander("摘要"):
                        st.write(paper.abstract)
                    
                    # Authors and venue
                    st.markdown(f"**作者:** {', '.join(paper.authors)}")
                    st.markdown(f"**发表场所:** {paper.venue}")
                    st.markdown(f"**年份:** {paper.year}")
                    st.markdown(f"**引用数:** {paper.citation_count}")
                    
                    # Bookmark button
                    paper_id = paper.title  # Use title as unique identifier
                    bookmark_key = f"bookmark_{paper_id}_{st.session_state.search_count}"
                    if st.button(f"⭐ Save", key=bookmark_key):
                        # Add paper to bookmarks
                        st.session_state.bookmarks[paper_id] = {
                            'title': paper.title,
                            'authors': paper.authors,
                            'year': paper.year,
                            'venue': paper.venue,
                            'url': paper.url,
                            'abstract': paper.abstract,
                            'citation_count': paper.citation_count,
                            'has_code': paper.has_code,
                            'is_top_venue': paper.is_top_venue,
                            'is_highly_cited': paper.is_highly_cited,
                            'is_open_access': paper.is_open_access
                        }
                        st.toast(f"Saved: {paper.title}")
