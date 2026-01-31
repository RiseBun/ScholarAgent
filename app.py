import streamlit as st
import sys
import os
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from backend.search_engine import PaperFetcher
from backend.tagger import enrich_and_tag
from backend.agent import parse_user_intent
from backend.landscape import LandscapeGenerator
from backend.quickscan import QuickScanner
from backend.scoop_checker import ScoopChecker
from backend.ideabreeder import IdeaBreeder
from backend.query_expander import QueryExpander
from backend.content_curator import ContentCurator

st.set_page_config(
    page_title="ScholarAgent - 科研猎手",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 ScholarAgent - 科研猎手")

# Main navigation
with st.sidebar:
    st.header("Navigation")
    app_mode = st.selectbox(
        "Select Mode",
        options=[
            "Quick Search",
            "Landscape Mapping",
            "Quick Scan",
            "Idea Breeder",
            "Scoop Checker"
        ],
        index=0
    )
    
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

# Quick Search Mode (existing functionality)
if app_mode == "Quick Search":
    with st.sidebar:
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
    
    # Initialize session state variables for pagination and query expansion
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
    
    # Display query input area - shows original or interpreted query
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
            analysis_status.info("Interpreting query...")
            
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
                
                analysis_status.success("Query interpretation completed!")
            except Exception as e:
                logger.error(f"Error interpreting query: {e}")
                analysis_status.error(f"Error interpreting query: {str(e)}")
    
    # Display user feedback window if query has been analyzed
    if st.session_state.query_analyzed:
        expanded_query = st.session_state.expanded_query
        
        with st.container(border=True):
            st.subheader("💬 Provide Feedback on AI's Interpretation")
            
            # Show current query and interpreted queries
            st.markdown(f"**Current Query:** {st.session_state.original_query}")
            st.markdown(f"**AI's Interpretation:** {st.session_state.interpreted_query}")
            st.markdown(f"**Extracted Venue:** {expanded_query.get('venue_filter', 'None')}")
            st.markdown(f"**Research Domain:** {expanded_query.get('domain', 'General')}")
            st.markdown(f"**Explanation:** {expanded_query.get('explanation', 'No explanation available')}")
            
            # User feedback input with better instructions
            user_feedback = st.text_area(
                "Your Feedback (please be specific about what's wrong or how to improve the interpretation)",
                value=st.session_state.user_feedback,
                height=150,
                key="user_feedback_input",
                placeholder="例如：\n- 我想要更关注最新的研究\n- 请包含更多相关的技术术语\n- 我的研究领域是计算机视觉，不是自然语言处理\n- 请不要包含不相关的venue"
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
    
    # Check if we need to display a historical page
    if 'current_papers' in st.session_state and 'current_expanded_query' in st.session_state:
        # Display historical page content
        expanded_query = st.session_state.current_expanded_query
        all_papers = st.session_state.current_papers
        
        # Clear the temporary session variables
        del st.session_state.current_papers
        del st.session_state.current_expanded_query
        
        # Display AI's interpretation of the query
        with st.expander("🤖 Final AI Query Interpretation"):
            st.markdown(f"**Original Query:** {user_query}")
            st.markdown(f"**Expanded Query:** {expanded_query.get('search_query', user_query)}")
            st.markdown(f"**Extracted Venue:** {expanded_query.get('venue_filter', 'None')}")
            st.markdown(f"**Research Domain:** {expanded_query.get('domain', 'General')}")
            st.markdown(f"**Explanation:** {expanded_query.get('explanation', 'No explanation available')}")
        
        # Display pagination info
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
                                    st.session_state.current_papers = h['papers']
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
        
        # Skip the rest of the code since we're displaying historical data
    elif (search_button or refresh_button) and st.session_state.user_approved:
        if not user_query:
            st.error("Please enter a research query")
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
            status_container.info(f"Searching papers with optimized strategy...")
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
                st.markdown(f"**Available Pages:** 1 - {st.session_state.current_page}")
                
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
                                        st.session_state.current_papers = h['papers']
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
            
            status_container.success(f"Search completed successfully! Found {len(all_papers)} papers on Page {st.session_state.current_page}.")
            logger.info(f"Search completed successfully! Found {len(all_papers)} papers on Page {st.session_state.current_page}.")

# Landscape Mapping Mode
elif app_mode == "Landscape Mapping":
    st.header("🌄 Research Landscape Mapping")
    st.info("Generate a panoramic view of your research field to identify hotspots, blue oceans, and research gaps.")
    
    user_query = st.text_area(
        "Enter your research field",
        placeholder="e.g., artificial intelligence, computer vision, robotics",
        height=100
    )
    
    with st.sidebar:
        st.header("Landscape Settings")
        paper_limit = st.slider("Number of Papers to Analyze", min_value=50, max_value=200, value=100, step=25)
    
    if st.button("Generate Landscape"):
        if not user_query:
            st.error("Please enter a research field")
        else:
            status_container = st.container()
            status_container.info("Generating research landscape... This may take a few minutes.")
            
            try:
                generator = LandscapeGenerator()
                landscape = generator.generate_landscape(user_query, llm_provider, api_key, limit=paper_limit)
                
                # Display results
                st.subheader(f"Research Landscape for: {user_query}")
                
                # Display visualization
                if landscape.get("visualization"):
                    st.image(f"data:image/png;base64,{landscape['visualization']}", caption="Research Landscape Visualization")
                
                # Display clusters
                st.subheader("Research Clusters")
                for cluster in landscape.get("clusters", []):
                    with st.container(border=True):
                        st.markdown(f"### Cluster {cluster['cluster_id']}: {cluster['cluster_type']}")
                        st.markdown(f"**Paper Count:** {cluster['paper_count']}")
                        st.markdown(f"**Average Citations:** {cluster['average_citations']:.1f}")
                        st.markdown(f"**Average Year:** {cluster['average_year']:.1f}")
                        st.markdown("**Sample Papers:**")
                        for title in cluster['sample_titles']:
                            st.markdown(f"- {title}")
                
                # Display subtopics
                st.subheader("Expanded Subtopics")
                st.write(landscape.get("subtopics", []))
                
                status_container.success(f"Landscape generated successfully! Analyzed {landscape.get('total_papers', 0)} papers.")
                
            except Exception as e:
                logger.error(f"Error generating landscape: {e}")
                status_container.error(f"Error generating landscape: {str(e)}")

# Quick Scan Mode
elif app_mode == "Quick Scan":
    st.header("⚡ Quick Paper Scan")
    st.info("Quickly scan multiple papers to extract core contributions, weaknesses, and future work.")
    
    user_query = st.text_area(
        "Enter your research query",
        placeholder="e.g., transformer models for NLP",
        height=100
    )
    
    with st.sidebar:
        st.header("Scan Settings")
        paper_limit = st.slider("Number of Papers to Scan", min_value=5, max_value=20, value=10, step=5)
    
    if st.button("Scan Papers"):
        if not user_query:
            st.error("Please enter a research query")
        else:
            status_container = st.container()
            status_container.info("Scanning papers...")
            
            try:
                # Search for papers
                fetcher = PaperFetcher()
                papers = fetcher.search_papers(user_query, limit=paper_limit)
                
                # Quick scan papers
                scanner = QuickScanner()
                analyzed_papers = scanner.quick_scan_papers(papers, llm_provider, api_key)
                
                # Display results
                st.subheader(f"Quick Scan Results ({len(analyzed_papers)} papers)")
                
                # Generate comparison table
                table_html = scanner.generate_comparison_table(analyzed_papers)
                st.markdown(table_html, unsafe_allow_html=True)
                
                # Display detailed analysis for each paper
                st.subheader("Detailed Analysis")
                for paper in analyzed_papers:
                    with st.expander(f"{paper['title']}"):
                        st.markdown(f"**Authors:** {', '.join(paper['authors'])}")
                        st.markdown(f"**Year:** {paper['year']}")
                        st.markdown(f"**Venue:** {paper['venue']}")
                        st.markdown(f"**Citations:** {paper['citation_count']}")
                        st.markdown(f"**URL:** [{paper['url']}]({paper['url']})")
                        st.markdown("**Core Contribution:**")
                        st.write(paper['core_contribution'])
                        st.markdown("**Weaknesses:**")
                        for weakness in paper['weaknesses']:
                            st.write(f"- {weakness}")
                        st.markdown("**Future Work:**")
                        for future in paper['future_work']:
                            st.write(f"- {future}")
                
                status_container.success(f"Quick scan completed successfully! Analyzed {len(analyzed_papers)} papers.")
                
            except Exception as e:
                logger.error(f"Error scanning papers: {e}")
                status_container.error(f"Error scanning papers: {str(e)}")

# Idea Breeder Mode
elif app_mode == "Idea Breeder":
    st.header("🧬 Idea Breeder")
    st.info("Breed and validate your research ideas through multi-agent debate and cross-disciplinary inspiration.")
    
    user_idea = st.text_area(
        "Describe your research idea",
        placeholder="e.g., Using transformer models for protein structure prediction",
        height=150
    )
    
    if st.button("Breed Idea"):
        if not user_idea:
            st.error("Please describe your research idea")
        else:
            status_container = st.container()
            status_container.info("Breeding and validating idea...")
            
            try:
                breeder = IdeaBreeder()
                result = breeder.breed_idea(user_idea, llm_provider, api_key)
                
                # Display results
                st.subheader("Idea Analysis")
                
                # Original vs Refined Idea
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Original Idea")
                    st.write(result['original_idea'])
                with col2:
                    st.markdown("### Refined Idea")
                    st.write(result['refined_idea'])
                
                # Debate
                st.subheader("Multi-Agent Debate")
                with st.container(border=True):
                    st.markdown("### Optimistic Perspective")
                    st.write(result['debate']['optimist'])
                with st.container(border=True):
                    st.markdown("### Critical Perspective")
                    st.write(result['debate']['critic'])
                
                # Cross-Breeding Suggestions
                st.subheader("Cross-Breeding Suggestions")
                for i, suggestion in enumerate(result['cross_breeding_suggestions'], 1):
                    st.markdown(f"**{i}.** {suggestion}")
                
                # Analogies
                if result.get('analogies'):
                    st.subheader("Analogous Research")
                    for analogy in result['analogies'][:3]:
                        with st.expander(f"{analogy['title']}"):
                            st.markdown(f"**Authors:** {', '.join(analogy['authors'])}")
                            st.markdown(f"**Year:** {analogy['year']}")
                            st.markdown(f"**Relevance:** {analogy['relevance']}")
                            st.markdown("**Abstract:**")
                            st.write(analogy['abstract'])
                
                # Feasibility Analysis
                st.subheader("Feasibility Analysis")
                feasibility = result['feasibility_analysis']
                st.markdown(f"**Feasibility Score:** {feasibility['feasibility_score']}/10")
                st.markdown("**Technical Analysis:**")
                st.write(feasibility['technical_analysis'])
                st.markdown("**Resource Estimation:**")
                st.write(feasibility['resource_estimation'])
                st.markdown("**Potential Roadblocks:**")
                for roadblock in feasibility['roadblocks']:
                    st.write(f"- {roadblock}")
                st.markdown("**Mitigation Strategies:**")
                for strategy in feasibility['mitigation_strategies']:
                    st.write(f"- {strategy}")
                
                status_container.success("Idea breeding completed successfully!")
                
            except Exception as e:
                logger.error(f"Error breeding idea: {e}")
                status_container.error(f"Error breeding idea: {str(e)}")

# Scoop Checker Mode
elif app_mode == "Scoop Checker":
    st.header("🔍 Scoop Checker")
    st.info("Check if your research idea has been scooped and get differentiation suggestions.")
    
    user_idea = st.text_area(
        "Describe your research idea in detail",
        placeholder="e.g., A new method for unsupervised domain adaptation using contrastive learning",
        height=150
    )
    
    with st.sidebar:
        st.header("Scoop Check Settings")
        paper_limit = st.slider("Number of Papers to Check", min_value=30, max_value=100, value=50, step=10)
    
    if st.button("Check for Scoops"):
        if not user_idea:
            st.error("Please describe your research idea")
        else:
            status_container = st.container()
            status_container.info("Checking for scoops... This may take a few minutes.")
            
            try:
                checker = ScoopChecker()
                result = checker.check_idea(user_idea, llm_provider, api_key, limit=paper_limit)
                
                # Display results
                st.subheader("Scoop Analysis")
                
                # Risk assessment
                risk_level = result['scoop_risk']
                if risk_level == "High":
                    st.error(f"🚨 Scoop Risk: {risk_level} (Score: {result['risk_score']:.2f})")
                elif risk_level == "Medium":
                    st.warning(f"⚠️ Scoop Risk: {risk_level} (Score: {result['risk_score']:.2f})")
                else:
                    st.success(f"✅ Scoop Risk: {risk_level} (Score: {result['risk_score']:.2f})")
                
                # Closest matches
                st.subheader("Closest Matching Papers")
                for i, match in enumerate(result['closest_matches'], 1):
                    with st.container(border=True):
                        st.markdown(f"### {i}. {match['title']}")
                        st.markdown(f"**Authors:** {', '.join(match['authors'])}")
                        st.markdown(f"**Year:** {match['year']}")
                        st.markdown(f"**Similarity Score:** {match['similarity_score']:.2f}")
                        st.markdown(f"**Venue:** {match['venue']}")
                        st.markdown(f"**Citations:** {match['citation_count']}")
                        st.markdown(f"**URL:** [{match['url']}]({match['url']})")
                        st.markdown("**Abstract:**")
                        st.write(match['abstract'][:300] + "..." if len(match['abstract']) > 300 else match['abstract'])
                
                # Differentiation suggestions
                st.subheader("Differentiation Suggestions")
                for i, suggestion in enumerate(result['differentiation_suggestions'], 1):
                    st.markdown(f"**{i}.** {suggestion}")
                
                # Gap analysis
                if result.get('gap_analysis'):
                    st.subheader("Gap Analysis")
                    st.write(result['gap_analysis'])
                
                status_container.success("Scoop check completed successfully!")
                
            except Exception as e:
                logger.error(f"Error checking for scoops: {e}")
                status_container.error(f"Error checking for scoops: {str(e)}")
