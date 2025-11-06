# app.py - Complete Streamlit frontend with Knowledge Graph
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging BEFORE any imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="M&A Document Intelligence",
    page_icon="📊",
    layout="wide"
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def check_api():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200, response.json()
    except:
        return False, None

# Check API Status
api_status, api_health = check_api()

if not api_status:
    st.error("⛔ API Backend not running!")
    st.info("Start the API with: `python api.py`")
    st.stop()

# Title
st.title("📊 M&A Document Intelligence Platform")

# Sidebar with API info and debugging
with st.sidebar:
    st.header("📌 API Status")
    st.success("✅ Connected to FastAPI")
    if api_health:
        st.metric("Documents", api_health['components']['documents'])
        st.metric("Chunks", api_health['components']['chunks'])
        if 'graph_nodes' in api_health['components']:
            st.metric("Graph Nodes", api_health['components']['graph_nodes'])
    st.caption(f"API: {API_BASE_URL}")
    st.caption(f"Docs: {API_BASE_URL}/docs")
    
    # Debug Section
    with st.expander("🔧 Debug Info"):
        st.write("### Environment Check")
        
        # Check API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        st.write("**API Key Status:**")
        if api_key:
            st.success(f"✅ Loaded ({api_key[:10]}...{api_key[-4:]})")
        else:
            st.error("❌ Not found in environment")
            
        st.write("**.env file:**", os.path.exists('.env'))
        st.write("**Working Dir:**", os.getcwd())
        
        # Test LLM directly
        if st.button("🧪 Test LLM Client"):
            with st.spinner("Testing LLM..."):
                try:
                    # Import and test LLM client
                    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                    from core.llm_client import LLMClient
                    
                    client = LLMClient()
                    st.write("Client initialized:", bool(client.client))
                    
                    if client.client:
                        # Try a simple generation
                        test_response = client.generate("Say 'hello'", max_tokens=10)
                        st.success(f"Response: {test_response}")
                    else:
                        st.error("Client not initialized - check API key")
                        
                        # Show what's in the environment
                        st.write("Environment variables starting with 'ANTHRO':")
                        for key, value in os.environ.items():
                            if key.startswith('ANTHRO'):
                                st.write(f"{key}: {value[:20]}...")
                                
                except Exception as e:
                    st.error(f"Test failed: {type(e).__name__}: {str(e)}")
                    import traceback
                    st.text(traceback.format_exc())

# Main tabs - UPDATED with Knowledge Graph tab
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📤 Upload",
    "🌐 Extract URL", 
    "💬 Q&A Analysis",
    "🔍 Search",
    "🗄️ Documents",
    "🕸️ Knowledge Graph"  # NEW!
])

# Upload Tab
with tab1:
    st.header("📤 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'xlsx', 'xls', 'docx', 'doc', 'eml'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("🚀 Upload to API", type="primary"):
            progress = st.progress(0)
            status_text = st.empty()
            
            files_data = []
            for file in uploaded_files:
                files_data.append(('files', (file.name, file.getvalue(), file.type)))
            
            try:
                status_text.text("Uploading... (OCR may take several minutes for scanned documents)")
                response = requests.post(
                    f"{API_BASE_URL}/api/documents/upload",
                    files=files_data,
                    timeout=600  # 10 minutes timeout for OCR processing
                )
                
                if response.status_code == 200:
                    results = response.json()
                    
                    success_count = sum(1 for r in results if r['status'] == 'success')
                    st.success(f"✅ Processed {success_count}/{len(results)} documents")
                    
                    # Show results in table
                    df = pd.DataFrame(results)
                    st.dataframe(df[['filename', 'status', 'word_count', 'chunks']])
                else:
                    st.error(f"Upload failed: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ Upload timed out. Large scanned PDFs may take longer to process. Please try uploading fewer files at once.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                progress.progress(1.0)
                status_text.empty()

# URL Extraction Tab
with tab2:
    st.header("🌐 Extract from URL")
    
    url = st.text_input("Enter URL:")
    extract_type = st.radio("Type", ["General Web Page", "SEC Filing"], horizontal=True)
    
    if st.button("🔗 Extract", disabled=not url):
        with st.spinner("Extracting..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/documents/extract-url",
                    json={"url": url, "extract_type": extract_type},
                    timeout=30  # 30 seconds for URL extraction
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ Extracted {result['word_count']} words")
                    st.info(f"Document ID: {result['doc_id']}")
                else:
                    st.error(response.json().get('detail', 'Extraction failed'))
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Q&A Analysis Tab
with tab3:
    st.header("💬 Intelligent Q&A Analysis")
    
    # Deal Analysis Section
    st.subheader("🎯 Automated Deal Analysis")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("Comprehensive M&A risk analysis across all documents (PDFs, Excel, URLs, Emails)")
    with col2:
        analyze_button = st.button("🚀 Analyze Deal", type="primary", use_container_width=True)
    
    # Create a placeholder for analysis results
    analysis_container = st.container()
    
    if analyze_button:
        with st.spinner("Analyzing all documents for deal-breaking issues..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/qa/analyze-deal",
                    timeout=120  # 2 minutes for deal analysis
                )
                
                if response.status_code == 200:
                    analysis = response.json()
                    
                    with analysis_container:
                        # Clear visual separation
                        st.markdown("---")
                        st.markdown("## 📊 Deal Analysis Results")
                        
                        # Score and Recommendation Display
                        score = analysis['deal_score']
                        rec = analysis['recommendation']
                        
                        # Create prominent score display using columns
                        score_col, rec_col = st.columns([1, 3])
                        
                        with score_col:
                            # Color-coded score display
                            if score >= 70:
                                st.markdown(f"""
                                <div style="background-color: #d4edda; border: 2px solid #28a745; 
                                border-radius: 10px; padding: 20px; text-align: center;">
                                <h1 style="color: #28a745; margin: 0;">{score}</h1>
                                <p style="margin: 0;">Deal Score</p>
                                </div>
                                """, unsafe_allow_html=True)
                            elif score >= 40:
                                st.markdown(f"""
                                <div style="background-color: #fff3cd; border: 2px solid #ffc107; 
                                border-radius: 10px; padding: 20px; text-align: center;">
                                <h1 style="color: #856404; margin: 0;">{score}</h1>
                                <p style="margin: 0;">Deal Score</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style="background-color: #f8d7da; border: 2px solid #dc3545; 
                                border-radius: 10px; padding: 20px; text-align: center;">
                                <h1 style="color: #dc3545; margin: 0;">{score}</h1>
                                <p style="margin: 0;">Deal Score</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with rec_col:
                            # Recommendation with appropriate styling
                            if "STOP" in rec:
                                st.error(f"### {rec}")
                            elif "CAUTION" in rec:
                                st.warning(f"### {rec}")
                            else:
                                st.success(f"### {rec}")
                        
                        # Summary Metrics
                        st.markdown("### 📈 Risk Summary")
                        metric_cols = st.columns(4)
                        with metric_cols[0]:
                            st.metric("Critical Issues", analysis['critical_count'], 
                                     delta="-" + str(analysis['critical_count']) if analysis['critical_count'] > 0 else "0",
                                     delta_color="inverse")
                        with metric_cols[1]:
                            st.metric("High Priority", analysis['high_count'],
                                     delta="-" + str(analysis['high_count']) if analysis['high_count'] > 0 else "0",
                                     delta_color="inverse")
                        with metric_cols[2]:
                            st.metric("Total Findings", analysis['total_issues'])
                        with metric_cols[3]:
                            st.metric("Analysis Time", f"{analysis.get('processing_time_ms', 0)}ms")
                        
                        # Detailed Findings
                        if analysis['findings']:
                            st.markdown("### 🔍 Detailed Findings")
                            
                            # Group findings by severity
                            critical_findings = [f for f in analysis['findings'] if f['severity'] == 'critical']
                            high_findings = [f for f in analysis['findings'] if f['severity'] == 'high']
                            medium_findings = [f for f in analysis['findings'] if f['severity'] == 'medium']
                            
                            # Display critical findings first
                            if critical_findings:
                                st.markdown("#### 🔴 Critical Issues (Deal Breakers)")
                                for finding in critical_findings:
                                    with st.expander(f"🚨 {finding['category'].replace('_', ' ').title()}", expanded=True):
                                        st.markdown(f"**Finding:** {finding['finding']}")
                                        if finding.get('sources'):
                                            st.markdown(f"**Sources:** {', '.join(finding['sources'][:3])}")
                                        st.markdown(f"**Confidence:** {finding.get('confidence', 0):.0%}")
                            
                            # High priority findings
                            if high_findings:
                                st.markdown("#### 🟡 High Priority Issues")
                                for finding in high_findings:
                                    with st.expander(f"⚠️ {finding['category'].replace('_', ' ').title()}"):
                                        st.markdown(f"**Finding:** {finding['finding']}")
                                        if finding.get('sources'):
                                            st.markdown(f"**Sources:** {', '.join(finding['sources'][:3])}")
                                        st.markdown(f"**Confidence:** {finding.get('confidence', 0):.0%}")
                            
                            # Medium priority findings
                            if medium_findings:
                                st.markdown("#### 🟠 Medium Priority Issues")
                                for finding in medium_findings:
                                    with st.expander(f"📋 {finding['category'].replace('_', ' ').title()}"):
                                        st.markdown(f"**Finding:** {finding['finding']}")
                                        if finding.get('sources'):
                                            st.markdown(f"**Sources:** {', '.join(finding['sources'][:3])}")
                                        st.markdown(f"**Confidence:** {finding.get('confidence', 0):.0%}")
                        
                        # Risk Matrix
                        if analysis.get('risk_matrix'):
                            st.markdown("### 🎯 Risk Distribution")
                            risk_df = pd.DataFrame(
                                list(analysis['risk_matrix'].items()),
                                columns=['Category', 'Count']
                            )
                            risk_df['Category'] = risk_df['Category'].str.replace('_', ' ').str.title()
                            st.bar_chart(risk_df.set_index('Category'))
                        
                        # Export option
                        st.markdown("---")
                        if st.button("📄 Export Report", type="secondary"):
                            st.info("Report export functionality coming soon!")
                    
                else:
                    error_detail = response.json() if response.content else "Unknown error"
                    st.error(f"Analysis failed: {error_detail}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.exception("Deal analysis error")
    
    # Custom Questions Section
    st.divider()
    st.subheader("💭 Ask Questions")
    
    # Add example questions
    example_questions = [
        "Custom question...",
        "What is the company's revenue in 2024?",
        "What are the main financial risks?",
        "Is there customer concentration risk?",
        "Are there any operational issues mentioned?",
        "What inconsistencies exist between documents?",
        "What is the company's debt level?",
        "Are there any regulatory compliance issues?",
        "What information is in the URL documents?"
    ]
    
    selected_example = st.selectbox("Example questions:", example_questions)
    
    if selected_example != "Custom question...":
        question = st.text_area("Your question:", value=selected_example, height=70)
    else:
        question = st.text_area("Your question:", placeholder="What are the main financial risks?", height=70)
    
    # Advanced options
    col1, col2 = st.columns([3, 1])
    with col1:
        show_sources = st.checkbox("Show detailed sources", value=True)
    with col2:
        show_debug = st.checkbox("Show debug info", value=False)
    
    if st.button("🔎 Get Answer", disabled=not question, type="primary"):
        with st.spinner("Searching all documents..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/qa/ask",
                    json={"question": question, "n_contexts": 5},
                    timeout=60  # 1 minute for Q&A
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Check if it's a system response
                    if result.get('is_system_response'):
                        st.info("### System Response")
                        st.write(result['answer'])
                    else:
                        # Check if relevant data was found
                        has_data = result.get('has_relevant_data', True)
                        confidence = result.get('confidence', 0)
                        
                        # Display answer with appropriate formatting
                        if not has_data or confidence < 0.4:
                            st.warning("### Answer")
                            st.write(result['answer'])
                            
                            # Suggest checking documents
                            if result.get('context_count', 0) == 0:
                                st.info("💡 **Tip**: Make sure you've uploaded relevant documents before asking questions.")
                        else:
                            st.success("### Answer")
                            st.write(result['answer'])
                        
                        # Show metadata in columns
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            if result.get('sources'):
                                st.caption(f"📚 Sources: {', '.join(result['sources'][:3])}")
                        with col2:
                            conf = result.get('confidence', 0)
                            conf_color = "🟢" if conf > 0.7 else "🟡" if conf > 0.4 else "🔴"
                            st.caption(f"{conf_color} Confidence: {conf:.0%}")
                        with col3:
                            st.caption(f"📄 Contexts: {result.get('context_count', 0)}")
                        
                        # Show detailed sources if enabled
                        if show_sources and result.get('source_details'):
                            st.divider()
                            st.subheader("📚 Source Documents")
                            
                            for i, source in enumerate(result.get('source_details', [])):
                                relevance = source.get('relevance', 0)
                                relevance_color = "🟢" if relevance > 0.7 else "🟡" if relevance > 0.4 else "🔴"
                                
                                with st.expander(f"{relevance_color} {source['filename']} (Relevance: {relevance:.0%})"):
                                    col1, col2 = st.columns([1, 3])
                                    with col1:
                                        st.write("**Type:**", source.get('doc_type', 'unknown'))
                                        st.write("**Chunk:**", source.get('chunk_id', 0) + 1)
                                    with col2:
                                        st.write("**Preview:**")
                                        st.text(source.get('preview', ''))
                    
                    # Show debug info if enabled
                    if show_debug:
                        with st.expander("🔧 Debug Information"):
                            st.json(result)
                            
                else:
                    error_detail = response.json() if response.content else "Unknown error"
                    st.error(f"Failed to get answer: {error_detail}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {str(e)}")
                st.info("Make sure the API server is running (python api.py)")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.exception("Q&A error")

# Search Tab
with tab4:
    st.header("🔍 Semantic Search")
    
    query = st.text_input("Search query:")
    
    if query:
        with st.spinner("Searching..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/search",
                    params={"query": query, "limit": 10},
                    timeout=30  # 30 seconds for search
                )
                
                if response.status_code == 200:
                    results = response.json()
                    
                    st.success(f"Found {len(results)} results")
                    
                    for i, result in enumerate(results):
                        with st.expander(f"Result {i+1} - {result['relevance']:.0%} match"):
                            st.write(result['text'])
                            st.caption(f"📄 {result['source']}")
                            st.caption(f"Chunk {result['chunk_info']['chunk_id']+1} of {result['chunk_info']['total_chunks']}")
                else:
                    st.error("Search failed")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Documents Tab  
with tab5:
    st.header("🗄️ Document Library")
    
    # Refresh button
    if st.button("🔄 Refresh"):
        st.rerun()
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/documents?limit=50", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            docs = data['documents']
            
            st.metric("Total Documents", data['pagination']['total'])
            
            if docs:
                # Create DataFrame for better display
                df = pd.DataFrame(docs)
                if 'upload_date' in df.columns and not df.empty:
                    df['upload_date'] = pd.to_datetime(df['upload_date']).dt.strftime('%Y-%m-%d %H:%M')
                
                st.dataframe(
                    df[['filename', 'doc_type', 'chunks', 'upload_date']],
                    use_container_width=True
                )
            else:
                st.info("No documents uploaded yet")
                
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")

# NEW: Knowledge Graph Tab
with tab6:
    st.header("🕸️ Knowledge Graph")
    
    # Check if graph is available
    try:
        graph_stats_response = requests.get(f"{API_BASE_URL}/api/graph/stats", timeout=5)
        
        if graph_stats_response.status_code == 200:
            graph_data = graph_stats_response.json()
            
            if graph_data.get('status') == 'available':
                stats = graph_data.get('stats', {})
                
                # Display stats
                st.subheader("📊 Graph Statistics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Nodes", stats.get('total_nodes', 0))
                with col2:
                    st.metric("Total Relationships", stats.get('total_relationships', 0))
                with col3:
                    st.metric("Node Types", len(stats.get('node_labels', {})))
                
                # Node breakdown
                if stats.get('node_labels'):
                    st.subheader("🏷️ Node Types")
                    node_df = pd.DataFrame(
                        list(stats['node_labels'].items()),
                        columns=['Type', 'Count']
                    )
                    st.bar_chart(node_df.set_index('Type'))
                
                # Relationship breakdown
                if stats.get('relationship_types'):
                    st.subheader("🔗 Relationship Types")
                    rel_df = pd.DataFrame(
                        list(stats['relationship_types'].items()),
                        columns=['Type', 'Count']
                    )
                    st.bar_chart(rel_df.set_index('Type'))
                
                # Build graph button
                st.divider()
                st.subheader("🏗️ Build Knowledge Graph")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write("Build knowledge graph from all uploaded documents")
                with col2:
                    if st.button("🚀 Build Graph", type="primary"):
                        with st.spinner("Building knowledge graph... This may take a minute."):
                            try:
                                build_response = requests.post(
                                    f"{API_BASE_URL}/api/graph/build-all",
                                    timeout=120
                                )
                                
                                if build_response.status_code == 200:
                                    result = build_response.json()
                                    
                                    st.success(f"✅ Graph built successfully!")
                                    st.write(f"**Documents processed:** {result['documents_processed']}")
                                    st.write(f"**Entities created:** {result['total_entities']}")
                                    st.write(f"**Relationships created:** {result['total_relationships']}")
                                    
                                    # Show details
                                    if result.get('details'):
                                        with st.expander("📝 Details"):
                                            details_df = pd.DataFrame(result['details'])
                                            st.dataframe(details_df)
                                    
                                    st.rerun()
                                else:
                                    st.error("Failed to build graph")
                                    
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                
                # Neo4j Browser link
                st.divider()
                st.subheader("🌐 View in Neo4j Browser")
                st.write("Open Neo4j Browser to explore the graph visually:")
                st.code("http://localhost:7474", language="text")
                st.caption("Login: neo4j / password123")
                
                # Sample queries
                with st.expander("📖 Sample Cypher Queries"):
                    st.code("""
# View all nodes
MATCH (n) RETURN n LIMIT 25

# Find organizations and their products
MATCH (o:Organization)-[:PRODUCES]->(p:Product)
RETURN o.name, p.name

# Find customer dependencies
MATCH (o1:Organization)-[r:DEPENDS_ON]->(o2:Organization)
RETURN o1.name, r.percentage, o2.name

# Find contradictions (CRITICAL!)
MATCH (n1)-[r:CONTRADICTS]->(n2)
RETURN n1, r, n2

# Find financial metrics by organization
MATCH (o:Organization)-[:HAS_METRIC]->(m:FinancialMetric)
RETURN o.name, m.name, m.value
ORDER BY m.value DESC

# Find all events and what they affect
MATCH (e:Event)-[:AFFECTS]->(target)
RETURN e.description, type(target), target.name

# Find products and their revenue
MATCH (p:Product)-[:GENERATES_REVENUE]->(m:FinancialMetric)
RETURN p.name, m.value
ORDER BY m.value DESC
                    """, language="cypher")
                
            elif graph_data.get('status') == 'unavailable':
                st.warning("⚠️ Knowledge Graph not initialized")
                st.info("Make sure Neo4j is running:")
                st.code("docker ps", language="bash")
                st.code("docker start neo4j-ma", language="bash")
            else:
                st.error(f"❌ Graph Error: {graph_data.get('message', 'Unknown error')}")
        else:
            st.error("❌ Could not connect to Knowledge Graph API")
            
    except Exception as e:
        st.error(f"Error loading graph: {str(e)}")
        st.info("Make sure Neo4j is running and the API can connect to it")

# Footer with instructions
st.divider()
st.caption("💡 Upload M&A documents → Build knowledge graph → Run deal analysis → Ask specific questions")