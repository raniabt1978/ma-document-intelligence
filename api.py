# api.py - Production FastAPI for M&A Document Intelligence with Knowledge Graph

# Environment variables to prevent warnings and issues
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Regular imports
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import asyncio
import io
import hashlib
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import all existing modules
from extractors.pdf_extractor import PDFExtractor
from extractors.excel_extractor import ExcelExtractor
from extractors.word_extractor import WordExtractor
from extractors.email_extractor import EmailExtractor
from extractors.web_extractor import WebExtractor
from core.vector_store import VectorStore
from core.document_processor import DocumentProcessor
from core.llm_client import LLMClient
from core.rag_engine import RAGEngine
from analyzers.financial_analyzer import FinancialAnalyzer
from analyzers.inconsistency_detector import InconsistencyDetector

# NEW: Knowledge Graph imports
from core.knowledge_graph_builder import KnowledgeGraphBuilder
from core.neo4j_client import get_neo4j_client

# Lifespan management for proper startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing M&A Document Intelligence API...")
    
    # Initialize all components
    app.state.vector_store = VectorStore()
    app.state.document_processor = DocumentProcessor()
    app.state.llm_client = LLMClient()
    app.state.rag_engine = RAGEngine(app.state.vector_store, app.state.llm_client)
    app.state.financial_analyzer = FinancialAnalyzer()
    app.state.inconsistency_detector = InconsistencyDetector()
    
    # NEW: Initialize Knowledge Graph Builder
    try:
        app.state.neo4j_client = get_neo4j_client()
        app.state.graph_builder = KnowledgeGraphBuilder(app.state.neo4j_client)
        logger.info("✅ Knowledge Graph Builder initialized")
    except Exception as e:
        logger.error(f"⚠️  Knowledge Graph initialization failed: {e}")
        app.state.graph_builder = None
    
    # Initialize extractors
    app.state.extractors = {
        'pdf': PDFExtractor(),
        'excel': ExcelExtractor(),
        'word': WordExtractor(),
        'email': EmailExtractor(),
        'web': WebExtractor()
    }
    
    logger.info("API initialized successfully")
    yield
    
    # Shutdown
    logger.info("Shutting down API...")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="M&A Document Intelligence API",
    description="""
    Production-grade document intelligence platform for M&A deal analysis.
    
    ## Core Capabilities
    - Multi-format document processing (PDF, Excel, Word, Email, Web)
    - OCR support for scanned documents
    - Hybrid RAG architecture with semantic search
    - Knowledge Graph for entity relationships
    - Financial health analysis and ratio calculations
    - Cross-document inconsistency detection
    - Automated deal-breaker identification
    
    ## Technical Stack
    - Vector Database: ChromaDB
    - Graph Database: Neo4j
    - LLM: Claude (Anthropic)
    - Processing: Async document pipeline
    - OCR: Tesseract + pdf2image
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],  # Streamlit and React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enums for validation
class ExtractType(str, Enum):
    general = "General Web Page"
    sec_filing = "SEC Filing"

class DocumentType(str, Enum):
    pdf = "pdf"
    excel = "excel"
    word = "word"
    email = "email"
    url = "url"

class SeverityLevel(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"

# Pydantic Models
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Question to ask about documents")
    n_contexts: int = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")
    filter_doc_types: Optional[List[DocumentType]] = Field(None, description="Filter by document types")

class URLExtractRequest(BaseModel):
    url: str = Field(..., pattern=r'^https?://', description="URL to extract content from")
    extract_type: ExtractType = Field(ExtractType.general, description="Type of extraction")

class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    doc_type: str
    status: str
    word_count: Optional[int]
    chunks: Optional[int]
    error: Optional[str]

class AnalysisResponse(BaseModel):
    deal_score: int = Field(..., ge=0, le=100)
    recommendation: str
    critical_count: int
    high_count: int
    total_issues: int
    findings: List[Dict[str, Any]]
    risk_matrix: Dict[str, int]
    processing_time_ms: int

class SearchResult(BaseModel):
    text: str
    relevance: float = Field(..., ge=0, le=1)
    source: str
    doc_type: str
    chunk_info: Dict[str, int]

# API Endpoints

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    try:
        stats = app.state.vector_store.get_stats()
        
        # Check graph status
        graph_status = "unavailable"
        graph_nodes = 0
        if app.state.graph_builder:
            try:
                graph_stats = app.state.graph_builder.get_graph_stats()
                graph_status = "operational"
                graph_nodes = graph_stats.get('total_nodes', 0)
            except:
                graph_status = "error"
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "vector_store": "operational",
                "knowledge_graph": graph_status,
                "llm_client": "operational" if app.state.llm_client.client else "degraded",
                "documents": stats['total_documents'],
                "chunks": stats['total_chunks'],
                "graph_nodes": graph_nodes
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/api/documents/upload", response_model=List[DocumentResponse], tags=["Documents"])
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Files to upload and process")
):
    """
    Upload and process multiple documents.
    Supports PDF, Excel, Word, Email formats.
    OCR is automatically applied to scanned PDFs.
    """
    results = []
    
    for file in files:
        start_time = datetime.utcnow()
        
        try:
            # Validate file type
            if not any(file.filename.lower().endswith(ext) for ext in ['.pdf', '.xlsx', '.xls', '.docx', '.doc', '.eml']):
                results.append(DocumentResponse(
                    doc_id="",
                    filename=file.filename,
                    doc_type="unknown",
                    status="error",
                    word_count=None,
                    chunks=None,
                    error="Unsupported file type"
                ))
                continue
            
            # Read file
            content = await file.read()
            file_obj = io.BytesIO(content)
            
            # Process document
            result = app.state.document_processor.process(file_obj, file.filename)
            
            # Generate unique doc_id
            doc_id = hashlib.md5(f"{file.filename}_{datetime.utcnow().isoformat()}".encode()).hexdigest()
            
            # Prepare metadata
            metadata = {
                "filename": file.filename,
                "doc_type": result['doc_type'],
                "file_size": len(content),
                "upload_date": datetime.utcnow().isoformat(),
                "processing_time_ms": int((datetime.utcnow() - start_time).total_seconds() * 1000)
            }
            
            # Add to vector store
            doc_id, is_new = app.state.vector_store.add_document(result['text'], metadata, doc_id=doc_id)
            
            if not is_new:
                logger.info(f"Document {file.filename} already exists, was replaced")
            
            # Calculate metrics
            word_count = len(result['text'].split())
            chunk_count = max(1, word_count // 500)  # Approximate chunks
            
            results.append(DocumentResponse(
                doc_id=doc_id,
                filename=file.filename,
                doc_type=result['doc_type'],
                status="success",
                word_count=word_count,
                chunks=chunk_count,
                error=None
            ))
            
            logger.info(f"Successfully processed {file.filename}")
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append(DocumentResponse(
                doc_id="",
                filename=file.filename,
                doc_type="unknown",
                status="error",
                word_count=None,
                chunks=None,
                error=str(e)
            ))
    
    return results

@app.post("/api/documents/extract-url", response_model=DocumentResponse, tags=["Documents"])
async def extract_from_url(request: URLExtractRequest):
    """Extract content from web URL or SEC filing"""
    start_time = datetime.utcnow()
    
    try:
        # Extract content based on type
        web_extractor = app.state.extractors['web']
        
        if request.extract_type == ExtractType.sec_filing:
            result = web_extractor.extract_sec_filing(request.url)
            text = result.get('full_text', '')
        else:
            text = web_extractor.extract_text(request.url)
        
        if not text or text.startswith("Error"):
            raise HTTPException(status_code=400, detail=text or "Failed to extract content")
        
        # Generate doc_id
        doc_id = hashlib.md5(request.url.encode()).hexdigest()
        
        # Add to vector store
        metadata = {
            "filename": request.url,
            "doc_type": "url",
            "url": request.url,
            "extract_type": request.extract_type.value,
            "upload_date": datetime.utcnow().isoformat()
        }
        
        doc_id, is_new = app.state.vector_store.add_document(text, metadata, doc_id=doc_id)
        
        return DocumentResponse(
            doc_id=doc_id,
            filename=request.url,
            doc_type="url",
            status="success",
            word_count=len(text.split()),
            chunks=max(1, len(text.split()) // 500),
            error=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents", tags=["Documents"])
async def list_documents(
    skip: int = Query(0, ge=0, description="Number of documents to skip"),
    limit: int = Query(20, ge=1, le=100, description="Maximum documents to return"),
    doc_type: Optional[DocumentType] = Query(None, description="Filter by document type")
):
    """List all documents with pagination and filtering"""
    all_data = app.state.vector_store.collection.get()
    
    # Group by document
    docs = {}
    for metadata in (all_data['metadatas'] or []):
        doc_id = metadata.get('doc_id', 'unknown')
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "filename": metadata.get('filename', 'Unknown'),
                "doc_type": metadata.get('doc_type', 'unknown'),
                "upload_date": metadata.get('upload_date', ''),
                "file_size": metadata.get('file_size', 0),
                "chunks": 1
            }
        else:
            docs[doc_id]['chunks'] += 1
    
    # Filter by type if requested
    doc_list = list(docs.values())
    if doc_type:
        doc_list = [d for d in doc_list if d['doc_type'] == doc_type.value]
    
    # Sort by upload date (newest first)
    doc_list.sort(key=lambda x: x['upload_date'], reverse=True)
    
    # Paginate
    total = len(doc_list)
    doc_list = doc_list[skip:skip + limit]
    
    return {
        "documents": doc_list,
        "pagination": {
            "total": total,
            "skip": skip,
            "limit": limit,
            "has_more": (skip + limit) < total
        }
    }

@app.post("/api/search", response_model=List[SearchResult], tags=["Search"])
async def search_documents(
    query: str = Query(..., min_length=1, max_length=500, description="Search query"),
    limit: int = Query(5, ge=1, le=20, description="Maximum results")
):
    """Semantic search across all documents"""
    results = app.state.vector_store.search(query, n_results=limit)
    
    return [
        SearchResult(
            text=r['text'][:500] + "..." if len(r['text']) > 500 else r['text'],
            relevance=1 - r['distance'],
            source=r['metadata'].get('filename', 'Unknown'),
            doc_type=r['metadata'].get('doc_type', 'unknown'),
            chunk_info={
                "chunk_id": r['metadata'].get('chunk_id', 0),
                "total_chunks": r['metadata'].get('total_chunks', 1)
            }
        )
        for r in results
    ]

@app.post("/api/qa/ask", tags=["Q&A"])
async def ask_question(request: QuestionRequest):
    """Ask questions about documents using RAG"""
    try:
        # Prepare filters if specified
        filters = None
        if request.filter_doc_types:
            filters = {"doc_type": {"$in": [dt.value for dt in request.filter_doc_types]}}
        
        # Get answer from RAG engine
        result = await app.state.rag_engine.answer_question_async(
            request.question,
            n_contexts=request.n_contexts,
            filters=filters
        )
        
        return {
            "question": request.question,
            "answer": result['answer'],
            "sources": result['sources'],
            "source_details": result.get('source_details', []),
            "confidence": result['confidence'],
            "context_count": result.get('context_count', 0),
            "processing_time_ms": result.get('processing_time_ms', 0),
            "has_relevant_data": result.get('has_relevant_data', True),
            "is_system_response": result.get('is_system_response', False)
        }
        
    except Exception as e:
        logger.error(f"Q&A error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/qa/analyze-deal", response_model=AnalysisResponse, tags=["Q&A"])
async def analyze_deal():
    """Run comprehensive M&A deal analysis"""
    try:
        # Check if we have documents
        stats = app.state.vector_store.get_stats()
        if stats['total_documents'] == 0:
            raise HTTPException(status_code=400, detail="No documents available for analysis")
        
        # Run analysis
        analysis = await app.state.rag_engine.analyze_deal_async()
        
        return AnalysisResponse(
            deal_score=analysis['deal_score'],
            recommendation=analysis['recommendation'],
            critical_count=analysis['critical_count'],
            high_count=analysis['high_count'],
            total_issues=analysis['total_issues'],
            findings=analysis['findings'],
            risk_matrix=analysis.get('risk_matrix', {}),
            processing_time_ms=analysis.get('processing_time_ms', 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deal analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analysis/financial/{doc_id}", tags=["Analysis"])
async def analyze_financial_health(doc_id: str):
    """Analyze financial health of specific document"""
    try:
        # Get document chunks
        chunks = app.state.vector_store.get_document_chunks(doc_id)
        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Combine text
        full_text = " ".join([chunk['text'] for chunk in chunks])
        
        # Run financial analysis
        analysis = app.state.financial_analyzer.analyze_financial_health(full_text)
        
        return {
            "doc_id": doc_id,
            "health_score": analysis['health_score'],
            "metrics_found": {
                metric_type: len(metrics) 
                for metric_type, metrics in analysis['metrics'].items() 
                if metrics
            },
            "ratios": [
                {
                    "name": ratio.name,
                    "value": ratio.value,
                    "is_concerning": ratio.is_concerning,
                    "concern_reason": ratio.concern_reason if ratio.is_concerning else None
                }
                for ratio in analysis['ratios']
            ],
            "concentration_risks": analysis['concentration_risks'],
            "summary": analysis['summary']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Financial analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# NEW: KNOWLEDGE GRAPH ENDPOINTS
# ============================================================================

@app.get("/api/graph/stats", tags=["Knowledge Graph"])
async def get_graph_stats():
    """Get knowledge graph statistics"""
    try:
        if not app.state.graph_builder:
            return {
                "status": "unavailable",
                "message": "Knowledge graph not initialized. Check Neo4j connection."
            }
        
        stats = app.state.graph_builder.get_graph_stats()
        
        return {
            "status": "available",
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/api/graph/build/{doc_id}", tags=["Knowledge Graph"])
async def build_graph_for_document(doc_id: str):
    """Build knowledge graph from a specific document"""
    try:
        if not app.state.graph_builder:
            raise HTTPException(
                status_code=503, 
                detail="Knowledge graph builder not available. Check Neo4j connection."
            )
        
        # Get document chunks
        chunks = app.state.vector_store.get_document_chunks(doc_id)
        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Combine text
        full_text = " ".join([chunk['text'] for chunk in chunks])
        
        # Get metadata from first chunk
        metadata = chunks[0]['metadata']
        
        # Build graph
        result = app.state.graph_builder.build_graph_from_document(full_text, metadata)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/graph/build-all", tags=["Knowledge Graph"])
async def build_graph_for_all_documents():
    """Build knowledge graph from all uploaded documents"""
    try:
        if not app.state.graph_builder:
            raise HTTPException(
                status_code=503,
                detail="Knowledge graph builder not available. Check Neo4j connection."
            )
        
        # Get all documents
        all_data = app.state.vector_store.collection.get()
        
        if not all_data['metadatas']:
            raise HTTPException(status_code=400, detail="No documents available")
        
        # Group by doc_id
        docs = {}
        for metadata in all_data['metadatas']:
            doc_id = metadata.get('doc_id', 'unknown')
            if doc_id not in docs:
                docs[doc_id] = {
                    'metadata': metadata,
                    'chunks': []
                }
        
        # Get chunks for each document
        results = []
        total_entities = 0
        total_relationships = 0
        
        for doc_id, doc_info in docs.items():
            try:
                # Get document chunks
                chunks = app.state.vector_store.get_document_chunks(doc_id)
                if not chunks:
                    continue
                
                # Combine text
                full_text = " ".join([chunk['text'] for chunk in chunks])
                metadata = chunks[0]['metadata']
                
                # Build graph
                result = app.state.graph_builder.build_graph_from_document(full_text, metadata)
                
                if result.get('status') == 'success':
                    entity_count = sum(result.get('entities_created', {}).values())
                    rel_count = result.get('relationships_created', 0)
                    
                    total_entities += entity_count
                    total_relationships += rel_count
                    
                    results.append({
                        'filename': metadata.get('filename'),
                        'entities': entity_count,
                        'relationships': rel_count
                    })
            except Exception as e:
                logger.error(f"Error processing document {doc_id}: {e}")
                continue
        
        return {
            "status": "success",
            "documents_processed": len(results),
            "total_entities": total_entities,
            "total_relationships": total_relationships,
            "details": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building graph for all documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/query", tags=["Knowledge Graph"])
async def query_graph(
    query: str = Query(..., description="Cypher query to execute"),
    limit: int = Query(25, ge=1, le=100, description="Result limit")
):
    """Execute Cypher query on knowledge graph"""
    try:
        if not app.state.graph_builder:
            raise HTTPException(
                status_code=503,
                detail="Knowledge graph not available"
            )
        
        # Execute query
        results = app.state.neo4j_client.execute_read(query)
        
        return {
            "query": query,
            "results": results[:limit],
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error executing graph query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SYSTEM ENDPOINTS
# ============================================================================

@app.get("/api/stats", tags=["System"])
async def get_statistics():
    """Get system statistics and metrics"""
    stats = app.state.vector_store.get_stats()
    
    # Get graph stats if available
    graph_stats = None
    if app.state.graph_builder:
        try:
            graph_stats = app.state.graph_builder.get_graph_stats()
        except:
            pass
    
    return {
        "vector_store": stats,
        "knowledge_graph": graph_stats,
        "api_version": "1.0.0",
        "llm_available": bool(app.state.llm_client.client),
        "graph_available": bool(app.state.graph_builder),
        "timestamp": datetime.utcnow().isoformat()
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting M&A Document Intelligence API...")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        timeout_keep_alive=600,  # 10 minutes keep-alive
        timeout_graceful_shutdown=30  # 30 seconds for graceful shutdown
    )