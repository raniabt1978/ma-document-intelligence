# core/rag_engine.py
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging
import re
from collections import defaultdict

from core.vector_store import VectorStore
from core.llm_client import LLMClient

logger = logging.getLogger(__name__)

@dataclass
class Finding:
    """Structured finding with metadata"""
    severity: str  # critical, high, medium, low
    category: str  # financial, operational, legal, strategic
    description: str
    sources: List[str]
    confidence: float
    evidence: str

class RAGEngine:
    """Production-ready RAG engine for M&A document analysis"""
    
    def __init__(self, vector_store: VectorStore, llm_client: LLMClient):
        self.vector_store = vector_store
        self.llm_client = llm_client
        
        # Sophisticated question templates for comprehensive analysis
        self.analysis_questions = {
            "customer_concentration": [
                "What percentage of revenue comes from the top customers?",
                "Is there customer concentration risk?",
                "How many customers account for majority of revenue?"
            ],
            "operational_continuity": [
                "Are there any mentions of stopping or discontinuing products or services?",
                "What operational changes are planned or required?",
                "Are there any regulatory or compliance issues mentioned?"
            ],
            "financial_health": [
                "What are the key financial metrics and ratios?",
                "How do inventory levels compare to revenue?",
                "What are the working capital requirements?"
            ],
            "strategic_alignment": [
                "Are there contradictions between different documents?",
                "Do growth projections align with operational reality?",
                "What strategic risks are identified?"
            ]
        }
        
        # Risk scoring weights
        self.severity_weights = {
            "critical": 30,
            "high": 15,
            "medium": 5,
            "low": 2
        }
        
        # System questions that shouldn't search documents
        self.system_questions = [
            'who are you', 'what are you', 'what can you do', 'hello', 'hi', 
            'help', 'how do you work', 'what is your purpose', 'introduce yourself'
        ]
        
        # Initialize cache for performance
        self._cache = {}
    
    async def answer_question_async(self, 
                                  question: str, 
                                  n_contexts: int = 5,
                                  filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Async question answering with intelligent context retrieval"""
        start_time = datetime.utcnow()
        
        # Check if this is a system question that shouldn't use documents
        question_lower = question.lower().strip()
        if any(sys_q in question_lower for sys_q in self.system_questions):
            return {
                "answer": "I'm an M&A Document Intelligence system designed to analyze merger and acquisition documents. I can help you identify financial risks, operational issues, customer concentration problems, and potential deal-breakers. Upload your M&A documents (PDFs, Excel files, emails, etc.) and ask me specific questions about them!",
                "sources": [],
                "source_details": [],
                "confidence": 1.0,
                "context_count": 0,
                "processing_time_ms": self._calculate_time(start_time),
                "is_system_response": True
            }
        
        # Check cache
        cache_key = f"{question}_{n_contexts}_{str(filters)}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            cached['from_cache'] = True
            return cached
        
        # Retrieve relevant contexts with semantic search
        contexts = self.vector_store.search(question, n_results=n_contexts, filter_metadata=filters)
        
        if not contexts:
            return {
                "answer": "I couldn't find relevant information in the uploaded documents to answer this question. Please make sure you've uploaded documents containing the information you're looking for.",
                "sources": [],
                "source_details": [],
                "confidence": 0.0,
                "context_count": 0,
                "processing_time_ms": self._calculate_time(start_time),
                "has_relevant_data": False
            }
        
        # Prepare enhanced context with metadata
        context_blocks = []
        sources = []
        source_details = []
        
        for i, ctx in enumerate(contexts):
            source = ctx['metadata'].get('filename', 'Unknown')
            doc_type = ctx['metadata'].get('doc_type', 'unknown')
            relevance = 1 - ctx['distance']
            chunk_id = ctx['metadata'].get('chunk_id', 0)
            
            context_blocks.append(
                f"[Document {i+1}: {source} | Type: {doc_type} | Relevance: {relevance:.2%}]\n{ctx['text']}"
            )
            
            source_detail = {
                "filename": source,
                "doc_type": doc_type,
                "relevance": relevance,
                "chunk_id": chunk_id,
                "preview": ctx['text'][:200] + "..." if len(ctx['text']) > 200 else ctx['text']
            }
            
            if source not in [s['filename'] for s in source_details]:
                source_details.append(source_detail)
                sources.append(source)
        
        context_text = "\n\n---\n\n".join(context_blocks)
        
        # Generate answer with temporal awareness
        prompt = self._create_temporal_aware_prompt(question, context_text)
        answer = await self.llm_client.generate_async(prompt, max_tokens=1500)
        
        # Check if answer indicates no relevant data found
        no_data_indicators = [
            "don't contain information", "no information available", 
            "documents do not include", "couldn't find", "not found in the documents",
            "no data for", "doesn't have information for", "no relevant data"
        ]
        
        answer_lower = answer.lower()
        has_relevant_data = not any(indicator in answer_lower for indicator in no_data_indicators)
        
        # Calculate confidence based on context relevance and answer quality
        avg_relevance = sum(1 - ctx['distance'] for ctx in contexts) / len(contexts)
        answer_confidence = self._assess_answer_confidence(answer, contexts, has_relevant_data)
        confidence = (avg_relevance * 0.6) + (answer_confidence * 0.4)
        
        result = {
            "answer": answer,
            "sources": sources[:5],  # Top 5 sources
            "source_details": source_details[:5],  # Detailed source info
            "confidence": confidence,
            "context_count": len(contexts),
            "processing_time_ms": self._calculate_time(start_time),
            "has_relevant_data": has_relevant_data
        }
        
        # Cache the result
        self._cache[cache_key] = result
        
        return result
    
    def answer_question(self, question: str, n_contexts: int = 5, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for compatibility"""
        return asyncio.run(self.answer_question_async(question, n_contexts, **kwargs))
    
    def _create_temporal_aware_prompt(self, question: str, context: str) -> str:
        """Create sophisticated analysis prompt with temporal awareness"""
        # Extract any years mentioned in the question
        years_in_question = re.findall(r'\b(19|20)\d{2}\b', question)
        
        temporal_instruction = ""
        if years_in_question:
            temporal_instruction = f"""
CRITICAL TEMPORAL REQUIREMENT:
- The user is asking specifically about the year(s): {', '.join(years_in_question)}
- ONLY provide information from that specific time period
- If the documents contain data from different years, you MUST:
  1. Clearly state what year the data is actually from
  2. Explicitly mention that you don't have data for the requested year
  3. Do NOT provide data from other years as if it answers the question
- Example: If asked about 2024 revenue but only have 2005 data, say: "I don't have 2024 revenue data in the documents. The documents show revenue of $X for 2005."
"""
        
        return f"""You are a senior M&A analyst conducting due diligence. Answer the question based ONLY on the provided document excerpts.
{temporal_instruction}

Question: {question}

Document Context:
{context[:4000]}

Instructions:
1. Answer based ONLY on the provided context
2. Always cite which specific document(s) you're getting information from
3. If asked about a specific year/date and the documents don't have that information, clearly state this
4. Be explicit about which year/time period any data is from
5. Never provide data from a different time period without clearly noting the year difference
6. Highlight ANY risks, concerns, or red flags
7. Be specific about percentages, dates, and figures
8. Note contradictions or inconsistencies

Format your answer to include document citations like: "According to [document name], ..."

Provide a clear, accurate answer that respects the temporal context of the question."""
    
    async def analyze_deal_async(self) -> Dict[str, Any]:
        """Comprehensive async deal analysis"""
        start_time = datetime.utcnow()
        findings = []
        
        # Parallel analysis across all categories
        tasks = []
        for category, questions in self.analysis_questions.items():
            for question in questions:
                task = self._analyze_question_async(question, category)
                tasks.append(task)
        
        # Execute all analyses in parallel
        results = await asyncio.gather(*tasks)
        
        # Process results into findings
        for result in results:
            if result:
                findings.append(result)
        
        # Calculate deal score and generate recommendation
        deal_score = self._calculate_deal_score(findings)
        recommendation = self._generate_recommendation(findings, deal_score)
        
        # Group findings by severity
        critical_findings = [f for f in findings if f.severity == "critical"]
        high_findings = [f for f in findings if f.severity == "high"]
        
        return {
            "deal_score": deal_score,
            "recommendation": recommendation,
            "critical_count": len(critical_findings),
            "high_count": len(high_findings),
            "total_issues": len(findings),
            "findings": [self._finding_to_dict(f) for f in findings],
            "risk_matrix": self._build_risk_matrix(findings),
            "processing_time_ms": self._calculate_time(start_time)
        }
    
    def quick_deal_analysis(self) -> Dict[str, Any]:
        """Synchronous wrapper for deal analysis"""
        return asyncio.run(self.analyze_deal_async())
    
    async def _analyze_question_async(self, question: str, category: str) -> Optional[Finding]:
        """Analyze a single question and create finding if issues detected"""
        result = await self.answer_question_async(question, n_contexts=10)
        
        # Skip system responses or low confidence results
        if result.get('is_system_response') or result['confidence'] < 0.3:
            return None
        
        # Analyze answer for risk indicators
        answer_lower = result['answer'].lower()
        severity = self._determine_severity(answer_lower)
        
        if severity in ['critical', 'high', 'medium']:
            return Finding(
                severity=severity,
                category=category,
                description=self._extract_key_finding(result['answer']),
                sources=result['sources'],
                confidence=result['confidence'],
                evidence=result['answer'][:500]
            )
        
        return None
    
    def _determine_severity(self, text: str) -> str:
        """Determine finding severity based on content analysis"""
        text_lower = text.lower()
        
        # Critical indicators
        critical_patterns = [
            'stop production', 'cease operation', 'discontinue',
            'bankruptcy', 'insolvency', 'default',
            '100%', '90%', '80%', 'sole customer', 'only customer',
            'material misstatement', 'fraud', 'investigation',
            'regulatory violation', 'compliance failure'
        ]
        
        # High severity indicators
        high_patterns = [
            'significant risk', 'major concern', 'substantial',
            '70%', '60%', 'concentration risk',
            'declining', 'deteriorating', 'negative trend',
            'litigation', 'dispute', 'conflict'
        ]
        
        # Check patterns
        if any(pattern in text_lower for pattern in critical_patterns):
            return "critical"
        elif any(pattern in text_lower for pattern in high_patterns):
            return "high"
        elif any(term in text_lower for term in ['risk', 'concern', 'issue', 'problem']):
            return "medium"
        else:
            return "low"
    
    def _extract_key_finding(self, answer: str) -> str:
        """Extract concise finding from detailed answer"""
        # Take first substantive sentence
        sentences = answer.split('.')
        for sentence in sentences:
            if len(sentence.strip()) > 20 and any(
                term in sentence.lower() 
                for term in ['risk', 'concern', 'issue', 'problem', 'found', 'identified']
            ):
                return sentence.strip() + '.'
        
        # Fallback to first sentence
        return sentences[0].strip() + '.' if sentences else answer[:200]
    
    def _calculate_deal_score(self, findings: List[Finding]) -> int:
        """Calculate deal score (0-100) based on findings"""
        if not findings:
            return 90  # No issues found
        
        # Start with perfect score
        score = 100
        
        # Deduct points based on findings
        for finding in findings:
            score -= self.severity_weights.get(finding.severity, 0)
        
        # Additional penalties for concentration
        concentration_findings = [f for f in findings if f.category == "customer_concentration"]
        if concentration_findings:
            score -= 10  # Extra penalty for any concentration risk
        
        return max(0, score)
    
    def _generate_recommendation(self, findings: List[Finding], score: int) -> str:
        """Generate actionable recommendation based on analysis"""
        critical_count = len([f for f in findings if f.severity == "critical"])
        high_count = len([f for f in findings if f.severity == "high"])
        
        if critical_count > 0:
            return "🔴 STOP DEAL: Critical issues identified that make this acquisition unviable without major restructuring"
        elif score < 40:
            return "🔴 HIGH RISK: Multiple severe issues require immediate attention - recommend against proceeding"
        elif score < 70:
            return "🟡 PROCEED WITH CAUTION: Significant concerns identified - renegotiate terms or implement risk mitigation"
        elif high_count > 2:
            return "🟡 MODERATE RISK: Several issues require investigation before proceeding"
        else:
            return "🟢 DEAL VIABLE: Standard due diligence findings - proceed with normal caution"
    
    def _build_risk_matrix(self, findings: List[Finding]) -> Dict[str, int]:
        """Build risk matrix by category"""
        matrix = defaultdict(int)
        for finding in findings:
            matrix[finding.category] += 1
        return dict(matrix)
    
    def _finding_to_dict(self, finding: Finding) -> Dict[str, Any]:
        """Convert Finding to dictionary for JSON serialization"""
        return {
            "severity": finding.severity,
            "category": finding.category,
            "finding": finding.description,
            "sources": finding.sources,
            "confidence": finding.confidence,
            "question": finding.category.replace("_", " ").title()
        }
    
    def _assess_answer_confidence(self, answer: str, contexts: List[Dict], has_relevant_data: bool) -> float:
        """Assess confidence of answer based on content and context quality"""
        if not answer or len(answer) < 50:
            return 0.3
        
        # Lower confidence if no relevant data found
        if not has_relevant_data:
            return 0.3
        
        # Check if answer contains specific data points
        has_numbers = any(char.isdigit() for char in answer)
        has_specific_mentions = any(
            term in answer.lower() 
            for term in ['specifically', 'states', 'indicates', 'shows', 'reveals', 'according to']
        )
        has_document_citations = any(
            term in answer.lower()
            for term in ['document', '.pdf', '.xlsx', '.doc', 'file', 'source']
        )
        
        confidence = 0.5
        if has_numbers:
            confidence += 0.15
        if has_specific_mentions:
            confidence += 0.15
        if has_document_citations:
            confidence += 0.1
        if len(contexts) >= 3:
            confidence += 0.1
            
        return min(confidence, 1.0)
    
    def _calculate_time(self, start_time: datetime) -> int:
        """Calculate processing time in milliseconds"""
        return int((datetime.utcnow() - start_time).total_seconds() * 1000)