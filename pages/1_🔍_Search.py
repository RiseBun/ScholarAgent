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
from backend.hybrid_search import HybridSearchEngine

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(
    page_title="ScholarAgent - Quick Search",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state variables (in case page is loaded directly)
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
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = 'openai'
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

st.title("🔍 Quick Search")

def diversify_papers(papers, top_n=20):
    """
    Diversify paper results using MMR (Maximal Marginal Relevance) algorithm
    
    Args:
        papers: List of paper dictionaries
        top_n: Number of papers to return
        
    Returns:
        Diversified list of papers
    """
    if len(papers) < top_n:
        return papers
    
    # Extract abstracts or titles for similarity calculation
    texts = []
    for paper in papers:
        if hasattr(paper, 'abstract') and paper.abstract:
            texts.append(paper.abstract)
        elif hasattr(paper, 'title') and paper.title:
            texts.append(paper.title)
        else:
            texts.append("")
    
    # Vectorize texts using TF-IDF
    try:
        tfidf = TfidfVectorizer(stop_words='english').fit_transform(texts)
        cosine_sim = cosine_similarity(tfidf)
        
        # MMR algorithm
        selected_indices = [0]  # Start with the most relevant paper
        candidate_indices = list(range(1, len(papers)))
        
        while len(selected_indices) < top_n and candidate_indices:
            # Calculate similarity of each candidate to selected papers
            similarities_to_selected = cosine_sim[candidate_indices][:, selected_indices]
            max_similarity_per_candidate = np.max(similarities_to_selected, axis=1)
            
            # Select candidate with minimum maximum similarity (most diverse)
            best_candidate_idx = np.argmin(max_similarity_per_candidate)
            original_idx = candidate_indices[best_candidate_idx]
            
            selected_indices.append(original_idx)
            candidate_indices.pop(best_candidate_idx)
            
        return [papers[i] for i in selected_indices]
    except Exception as e:
        logger.error(f"Error in diversify_papers: {e}")
        # Fallback to original list if diversification fails
        return papers[:top_n]

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
col1 = st.columns([2])[0]
with col1:
    analyze_button = st.button("🤖 Interpret Query")

# Process feedback and analysis
# This function handles both initial analysis and feedback-based reanalysis
def process_analysis():
    if not user_query:
        st.error("Please enter a research query")
        return False
    
    # Always update original query to current input
    st.session_state.original_query = user_query
    
    # Reset states for new analysis
    st.session_state.user_approved = False
    
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
        return True
    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        analysis_status.error(f"Error analyzing query: {str(e)}")
        return False

# Interpret query when analyze button is clicked
if analyze_button:
    process_analysis()

# Handle feedback-based reanalysis
if st.session_state.feedback_given:
    # If feedback was given, process it regardless of analyze_button state
    # This handles the case after st.rerun() from Update Interpretation button
    process_analysis()
    # Reset feedback_given after processing to avoid infinite loops
    st.session_state.feedback_given = False

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
        
        # Update session state with current feedback input
        if user_feedback != st.session_state.user_feedback:
            st.session_state.user_feedback = user_feedback
        
        # Feedback action buttons
        feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
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
                st.success("Interpretation approved! Starting new search with updated parameters...")
                # Reset pagination for new search
                st.session_state.current_page = 1
                st.session_state.search_history = []
                # Force a rerun to update the button state and trigger search
                st.rerun()
        
        with feedback_col3:
            if st.button("🔄 Change Batch"):
                if st.session_state.user_approved:
                    # Increment batch counter to get different papers
                    st.session_state.batch_counter = st.session_state.get('batch_counter', 0) + 1
                    st.session_state.batch_change = True
                    st.info(f"Loading Batch {st.session_state.batch_counter + 1}...")
                    st.rerun()
                else:
                    st.warning("Please approve interpretation first")
        
        # Show batch navigation if we have multiple batches
        if st.session_state.get('batch_counter', 0) > 0:
            st.markdown(f"**Current Batch:** {st.session_state.get('batch_counter', 0) + 1}")

# Track if search was just performed to avoid duplicate display
search_performed_this_run = False

# Trigger search if batch change button is clicked, or if interpretation was just approved
if ((st.session_state.get('batch_change', False)) or \
    (st.session_state.user_approved and 'expanded_query' in st.session_state and \
     ('last_expanded_query' not in st.session_state or \
      st.session_state.last_expanded_query != st.session_state.expanded_query))) and \
   st.session_state.user_approved:
    if not user_query:
        st.error("Please enter a research query")
    else:
        # Determine if it's a new search or batch change
        current_expanded_query = st.session_state.expanded_query.get('search_query', user_query)
        last_expanded_query = st.session_state.get('last_expanded_query', '')
        
        is_new_search = user_query != st.session_state.last_query or current_expanded_query != last_expanded_query
        is_batch_change = st.session_state.get('batch_change', False)
        
        if is_new_search:
            # Reset for new search
            st.session_state.current_page = 1
            st.session_state.batch_counter = 0
            st.session_state.search_history = []
            st.session_state.last_query = user_query
            st.session_state.last_expanded_query = current_expanded_query
        elif is_batch_change:
            # For batch change, increment page and use batch_counter for offset
            batch_num = st.session_state.get('batch_counter', 0)
            st.session_state.current_page = batch_num + 1
        
        # Reset batch_change flag after processing
        st.session_state.batch_change = False
        
        # Calculate offset based on batch counter (each batch gets different papers)
        batch_counter = st.session_state.get('batch_counter', 0)
        offset = batch_counter * result_limit  # Each batch skips previous batches' papers
        
        # Increment search count to force refresh
        st.session_state.search_count += 1
        
        # Create a status container to display real-time updates with progress bar
        progress_container = st.container()
        progress_bar = progress_container.progress(0, text="Starting search...")
        status_text = progress_container.empty()
        
        def update_progress(percent: int, message: str):
            """Helper to update progress bar and status text"""
            progress_bar.progress(percent, text=message)
            status_text.info(message)
        
        update_progress(5, "Initializing search engine...")
        
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
        
        # Use batch offset for getting different papers (already calculated above)
        # offset is set based on batch_counter to skip previously shown papers
        
        # Use optimized search strategy with expanded query
        update_progress(15, f"Searching papers (Batch {batch_counter + 1})...")
        logger.info(f"Searching papers with optimized strategy for query: '{expanded_query.get('search_query', user_query)}', offset: {offset}")
        
        try:
            # Use hybrid search engine
            hybrid_engine = HybridSearchEngine()
            
            # Get venue filter from expanded query
            venue_filter = expanded_query.get('venue_filter', '')
            
            update_progress(25, "Using hybrid search: LLM recommendations + database search...")
            
            # Perform hybrid search
            search_result = hybrid_engine.search_sync(
                user_query=user_query,
                llm_provider=llm_provider,
                api_key=api_key,
                limit=result_limit,
                year_range=year_range,
                venue_filter=venue_filter,
                offset=offset
            )
            
            # Extract results
            llm_recommended = search_result.get('llm_recommended', [])
            db_discovered = search_result.get('db_discovered', [])
            merged_papers = search_result.get('merged', [])
            metadata = search_result.get('metadata', {})
            
            logger.info(f"Hybrid search: LLM recommended {len(llm_recommended)}, DB discovered {len(db_discovered)}")
            logger.info(f"Metadata: {metadata}")
            
            update_progress(70, f"Found {len(merged_papers)} papers, applying filters...")
            
            # Store search result and metadata in session state
            st.session_state.llm_recommended_papers = llm_recommended
            st.session_state.db_discovered_papers = db_discovered
            st.session_state.search_metadata = metadata
            st.session_state.search_result = search_result  # Store full result for rerank reasoning
            
            # Apply filters to merged papers
            for paper in merged_papers:
                # Apply filters
                if (not must_have_code or paper.has_code) and \
                   (not top_venue_only or paper.is_top_venue) and \
                   (not highly_cited_only or paper.is_highly_cited):
                    all_papers.append(paper)
                    displayed_count += 1
            
            # If still no papers after filtering, use all merged
            if not all_papers and merged_papers:
                update_progress(80, "No papers passed filters, showing all results...")
                all_papers = merged_papers
            
            # Diversify results to improve variety
            if all_papers:
                update_progress(85, "Diversifying results for better variety...")
                all_papers = diversify_papers(all_papers, top_n=result_limit)
                
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            progress_bar.progress(50, text="Error occurred, trying fallback...")
            status_text.error(f"Error searching papers: {str(e)}")
            # Fallback to traditional search
            update_progress(55, "Falling back to traditional search...")
            try:
                fetcher = PaperFetcher()
                search_query = expanded_query.get('search_query', user_query)
                venue_filter = expanded_query.get('venue_filter', '')
                papers = fetcher.search_papers(
                    search_query, 
                    limit=result_limit, 
                    year_range=year_range, 
                    venue_filter=venue_filter,
                    offset=offset
                )
                for paper_data in papers:
                    tagged_papers = enrich_and_tag([paper_data])
                    if tagged_papers:
                        all_papers.append(tagged_papers[0])
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
        
        # Save current batch to history
        st.session_state.search_history.append({
            'batch': batch_counter + 1,
            'page': st.session_state.current_page,
            'query': user_query,
            'expanded_query': expanded_query,
            'papers': all_papers,
            'offset': offset
        })
        
        # Store results in session state
        st.session_state.current_results = all_papers
        
        # Display AI's interpretation of the query (now above results)
        with st.expander("Final AI Query Interpretation"):
            st.markdown(f"**Original Query:** {user_query}")
            st.markdown(f"**Expanded Query:** {expanded_query.get('search_query', user_query)}")
            st.markdown(f"**Extracted Venue:** {expanded_query.get('venue_filter', 'None')}")
            st.markdown(f"**Research Domain:** {expanded_query.get('domain', 'General')}")
            st.markdown(f"**Explanation:** {expanded_query.get('explanation', 'No explanation available')}")
        
        # Display hybrid search metadata
        if 'search_metadata' in st.session_state:
            meta = st.session_state.search_metadata
            with st.expander("Search Statistics", expanded=True):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("AI Selected", meta.get('merged_count', 0))
                with col2:
                    st.metric("DB Found", meta.get('db_search_count', 0))
                with col3:
                    st.metric("Venue Matched", meta.get('venue_matched_count', 0))
                with col4:
                    rerank_status = "Yes" if meta.get('rerank_enabled', False) else "No"
                    st.metric("AI Reranked", rerank_status)
                with col5:
                    venue = meta.get('venue_filter', '')
                    st.metric("Target Venue", venue if venue else "Any")
        
        # Display AI reranking reasoning (if available)
        if 'search_result' in st.session_state:
            sr = st.session_state.search_result
            if sr.get('rerank_reasoning'):
                with st.expander("AI Selection Strategy", expanded=True):
                    st.info(f"**AI's reasoning:** {sr['rerank_reasoning']}")
        
        # Display batch navigation info
        current_batch = st.session_state.get('batch_counter', 0) + 1
        st.markdown(f"**Current Batch:** {current_batch}")
        if len(st.session_state.search_history) > 1:
            st.markdown(f"**Available Batches:** {len(st.session_state.search_history)}")
            
            # Add buttons to navigate to previous batches
            with st.expander("📋 View Previous Batches"):
                for idx, history in enumerate(st.session_state.search_history):
                    batch_num = history.get('batch', idx + 1)
                    if batch_num != current_batch:
                        if st.button(f"View Batch {batch_num} ({len(history['papers'])} papers)", key=f"batch_nav_{batch_num}"):
                            st.session_state.batch_counter = batch_num - 1
                            st.session_state.current_page = batch_num
                            st.session_state.current_results = history['papers']
                            st.session_state.current_expanded_query = history['expanded_query']
                            st.rerun()
        
        # Update results header
        update_progress(90, "Preparing results display...")
        if all_papers:
            results_header.subheader(f"Results (Batch {current_batch}) - Found {len(all_papers)} papers")
        else:
            results_header.subheader(f"Results (Batch {current_batch}) - No papers found")
            status_text.warning("No papers found. Showing most relevant results...")
        
        # Display papers now
        for idx, paper in enumerate(all_papers):
            # Create a new container for this paper
            paper_container = results_container.empty()
            paper_containers.append(paper_container)
            
            # Display the new paper
            with paper_container:
                with st.container(border=True):
                    # Title with link and badges
                    title_col, score_col, badge_col = st.columns([3, 1, 1])
                    with title_col:
                        # Add "Must Read" prefix for baseline papers
                        is_baseline = getattr(paper, 'is_baseline', False)
                        title_prefix = "⭐ " if is_baseline else ""
                        if paper.url:
                            st.markdown(f"### {title_prefix}[{paper.title}]({paper.url})")
                        else:
                            st.markdown(f"### {title_prefix}{paper.title}")
                    with score_col:
                        # Show quality score or LLM relevance score
                        quality_score = getattr(paper, 'quality_score', 0)
                        llm_relevance = getattr(paper, 'llm_relevance_score', 0)
                        display_score = llm_relevance if llm_relevance > 0 else quality_score
                        if display_score > 0:
                            if display_score >= 70:
                                st.success(f"Score: {display_score:.0f}")
                            elif display_score >= 50:
                                st.info(f"Score: {display_score:.0f}")
                            else:
                                st.warning(f"Score: {display_score:.0f}")
                    with badge_col:
                        # Show source badge based on paper source
                        source = getattr(paper, 'source', '')
                        if is_baseline:
                            st.error("Must Read")
                        elif source == 'llm_reranked':
                            st.success("AI Selected")
                        elif source == 'llm_verified':
                            st.success("LLM Verified")
                        elif paper.is_top_venue:
                            st.info("Top Venue")
                    
                    # Tags row
                    tags = []
                    if is_baseline:
                        tags.append("Must Read")
                    if paper.is_top_venue:
                        tags.append("Top Venue")
                    if paper.has_code:
                        tags.append("Has Code")
                    if paper.is_highly_cited:
                        tags.append("Highly Cited")
                    if paper.is_open_access:
                        tags.append("Open Access")
                    
                    if tags:
                        st.markdown(f"**Tags:** {' | '.join(tags)}")
                    
                    # LLM recommendation reason (for reranked papers)
                    llm_reasoning = getattr(paper, 'llm_reasoning', '')
                    if llm_reasoning and source in ['llm_reranked', 'llm_verified']:
                        st.info(f"**Why recommended:** {llm_reasoning}")
                    
                    # Score breakdown (if available)
                    score_breakdown = getattr(paper, 'score_breakdown', None)
                    if score_breakdown:
                        with st.expander("Quality Score Breakdown"):
                            breakdown_cols = st.columns(4)
                            with breakdown_cols[0]:
                                st.metric("Relevance", f"{score_breakdown.get('relevance', 0):.0f}")
                            with breakdown_cols[1]:
                                st.metric("Citation", f"{score_breakdown.get('citation', 0):.0f}")
                            with breakdown_cols[2]:
                                st.metric("Venue", f"{score_breakdown.get('venue', 0):.0f}")
                            with breakdown_cols[3]:
                                st.metric("Recency", f"{score_breakdown.get('recency', 0):.0f}")
                    
                    # Abstract (expandable)
                    with st.expander("Abstract"):
                        st.write(paper.abstract)
                    
                    # Paper metadata
                    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
                    with meta_col1:
                        st.markdown(f"**Venue:** {paper.venue if paper.venue else 'Unknown'}")
                    with meta_col2:
                        st.markdown(f"**Year:** {paper.year if paper.year else 'Unknown'}")
                    with meta_col3:
                        st.markdown(f"**Citations:** {paper.citation_count}")
                    with meta_col4:
                        # Bookmark button
                        paper_id = f"{idx}_{paper.title}"
                        bookmark_key = f"bookmark_search_{paper_id}_{st.session_state.search_count}"
                        if st.button(f"Save", key=bookmark_key):
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
                                'is_open_access': paper.is_open_access,
                                'source': source,
                                'quality_score': quality_score
                            }
                            st.toast(f"Saved: {paper.title}")
        
        # Update last expanded query to detect changes
        st.session_state.last_expanded_query = st.session_state.expanded_query
        
        # Mark that search was performed this run
        search_performed_this_run = True
        
        # Complete progress bar
        progress_bar.progress(100, text=f"Search completed! Found {len(all_papers)} papers")
        status_text.success(f"Search completed! Found {len(all_papers)} papers in Batch {current_batch}.")
        logger.info(f"Search completed! Found {len(all_papers)} papers in Batch {current_batch}.")

# Display historical results if available (but not if search was just performed)
if 'current_results' in st.session_state and st.session_state.current_results:
    if not analyze_button and not st.session_state.get('batch_change', False) and not search_performed_this_run:
        # Display historical results
        expanded_query = st.session_state.get('current_expanded_query', st.session_state.expanded_query)
        all_papers = st.session_state.current_results
        
        # Display AI's interpretation of the query
        with st.expander("AI Query Interpretation"):
            st.markdown(f"**Original Query:** {st.session_state.original_query}")
            st.markdown(f"**Expanded Query:** {expanded_query.get('search_query', st.session_state.original_query)}")
            st.markdown(f"**Extracted Venue:** {expanded_query.get('venue_filter', 'None')}")
            st.markdown(f"**Research Domain:** {expanded_query.get('domain', 'General')}")
            st.markdown(f"**Explanation:** {expanded_query.get('explanation', 'No explanation available')}")
        
        # Display search metadata (if available)
        if 'search_metadata' in st.session_state:
            meta = st.session_state.search_metadata
            with st.expander("Search Statistics"):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("LLM Verified", meta.get('llm_verified_count', 0))
                with col2:
                    st.metric("DB Found", meta.get('db_search_count', 0))
                with col3:
                    st.metric("Venue Matched", meta.get('venue_matched_count', 0))
                with col4:
                    st.metric("Total Results", meta.get('merged_count', 0))
                with col5:
                    venue = meta.get('venue_filter', '')
                    st.metric("Target Venue", venue if venue else "Any")
        
        # Display pagination info
        st.markdown(f"**Current Page:** {st.session_state.current_page}")
        if len(st.session_state.search_history) > 1:
            st.markdown(f"**Available Pages:** 1 - {len(st.session_state.search_history)}")
            
            # Add buttons to navigate to previous pages
            with st.expander("View History Pages"):
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
        results_container = st.container()
        results_header = results_container.empty()
        if all_papers:
            results_header.subheader(f"Results (Page {st.session_state.current_page}) - Found {len(all_papers)} papers")
        else:
            results_header.subheader(f"Results (Page {st.session_state.current_page}) - No papers found")
        
        # Display papers
        for idx, paper in enumerate(all_papers):
            # Create a new container for this paper
            paper_container = results_container.empty()
            
            # Display the paper
            with paper_container:
                with st.container(border=True):
                    # Title with link and source badge
                    title_col, score_col, badge_col = st.columns([3, 1, 1])
                    with title_col:
                        if paper.url:
                            st.markdown(f"### [{paper.title}]({paper.url})")
                        else:
                            st.markdown(f"### {paper.title}")
                    with score_col:
                        # Show quality score
                        quality_score = getattr(paper, 'quality_score', 0)
                        if quality_score > 0:
                            if quality_score >= 70:
                                st.success(f"Score: {quality_score:.0f}")
                            elif quality_score >= 50:
                                st.info(f"Score: {quality_score:.0f}")
                            else:
                                st.warning(f"Score: {quality_score:.0f}")
                    with badge_col:
                        # Show source badge - only for LLM verified papers
                        source = getattr(paper, 'source', '')
                        if source == 'llm_verified':
                            st.success("LLM Verified")
                        elif paper.is_top_venue:
                            st.info("Top Venue")
                    
                    # Tags row
                    tags = []
                    if paper.is_top_venue:
                        tags.append("Top Venue")
                    if paper.has_code:
                        tags.append("Has Code")
                    if paper.is_highly_cited:
                        tags.append("Highly Cited")
                    if paper.is_open_access:
                        tags.append("Open Access")
                    
                    if tags:
                        st.markdown(f"**Tags:** {' | '.join(tags)}")
                    
                    # Score breakdown (if available)
                    score_breakdown = getattr(paper, 'score_breakdown', None)
                    if score_breakdown:
                        with st.expander("Quality Score Breakdown"):
                            breakdown_cols = st.columns(4)
                            with breakdown_cols[0]:
                                st.metric("Relevance", f"{score_breakdown.get('relevance', 0):.0f}")
                            with breakdown_cols[1]:
                                st.metric("Citation", f"{score_breakdown.get('citation', 0):.0f}")
                            with breakdown_cols[2]:
                                st.metric("Venue", f"{score_breakdown.get('venue', 0):.0f}")
                            with breakdown_cols[3]:
                                st.metric("Recency", f"{score_breakdown.get('recency', 0):.0f}")
                    
                    # LLM reasoning (only if from LLM and has reasoning)
                    llm_reasoning = getattr(paper, 'llm_reasoning', '')
                    if source == 'llm_verified' and llm_reasoning:
                        with st.expander("LLM Recommendation Reason"):
                            st.write(llm_reasoning)
                    
                    # Abstract (expandable)
                    with st.expander("Abstract"):
                        st.write(paper.abstract)
                    
                    # Paper metadata
                    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
                    with meta_col1:
                        st.markdown(f"**Venue:** {paper.venue if paper.venue else 'Unknown'}")
                    with meta_col2:
                        st.markdown(f"**Year:** {paper.year if paper.year else 'Unknown'}")
                    with meta_col3:
                        st.markdown(f"**Citations:** {paper.citation_count}")
                    with meta_col4:
                        # Bookmark button
                        paper_id = f"{idx}_{paper.title}"
                        bookmark_key = f"bookmark_history_{paper_id}_{st.session_state.search_count}"
                        if st.button(f"Save", key=bookmark_key):
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
                                'is_open_access': paper.is_open_access,
                                'source': source,
                                'quality_score': quality_score
                            }
                            st.toast(f"Saved: {paper.title}")
