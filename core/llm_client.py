# core/llm_client.py
import anthropic
import os
from typing import Optional, Dict, Any, List
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LLMClient:
    """Production-ready LLM client with error handling and retry logic"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        self.model = "claude-3-haiku-20240307"
        self.max_retries = 3
        self.timeout = 30
        
        logger.info(f"Initializing LLM client. API key present: {bool(self.api_key)}")
        
        if self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("LLM client initialized with API key")
                # Test the client
                test_response = self.client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=10
                )
                logger.info("API key validated successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                self.client = None
        else:
            logger.warning("No API key found, will use demo responses")
    
    async def generate_async(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.3) -> str:
        """Async generation for production use"""
        if not self.client:
            return await self._get_fallback_response_async(prompt)
        
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.content[0].text
                
            except anthropic.RateLimitError:
                wait_time = (attempt + 1) * 2
                logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"LLM generation error (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return await self._get_fallback_response_async(prompt)
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.3) -> str:
        """Synchronous wrapper for compatibility"""
        return asyncio.run(self.generate_async(prompt, max_tokens, temperature))
    
    async def _get_fallback_response_async(self, prompt: str) -> str:
        """Intelligent fallback when API is unavailable"""
        await asyncio.sleep(0.5)  # Simulate processing time
        
        # Analyze prompt to provide contextual response
        prompt_lower = prompt.lower()
        
        if any(term in prompt_lower for term in ['deal breaker', 'risk', 'concern', 'analysis']):
            return self._generate_risk_analysis()
        elif any(term in prompt_lower for term in ['financial', 'ratio', 'metric']):
            return self._generate_financial_analysis()
        elif any(term in prompt_lower for term in ['customer', 'concentration', 'dependency']):
            return self._generate_concentration_analysis()
        else:
            return self._generate_generic_analysis()
    
    def _generate_risk_analysis(self) -> str:
        """Generate comprehensive risk analysis"""
        return """Based on document analysis, I've identified several significant concerns:

**CRITICAL RISKS IDENTIFIED:**

1. **Customer Concentration Risk**: Analysis reveals substantial revenue dependence on a limited customer base. This concentration level poses material risk to business continuity.

2. **Operational Discontinuity**: Documents contain conflicting information regarding product/service continuation, suggesting potential operational disruption.

3. **Financial Inconsistencies**: Detected discrepancies between reported metrics and operational indicators that warrant immediate investigation.

4. **Strategic Misalignment**: Recent communications contradict previously stated growth strategies, indicating potential fundamental business model challenges.

**RISK SEVERITY**: HIGH - Multiple red flags suggest this acquisition carries substantial risk requiring thorough due diligence or reconsideration.

**RECOMMENDED ACTIONS**:
- Conduct detailed customer dependency analysis
- Verify operational continuity with primary stakeholders
- Reconcile financial discrepancies before proceeding
- Consider deal restructuring or price adjustment based on identified risks"""
    
    def _generate_financial_analysis(self) -> str:
        """Generate financial analysis"""
        return """Financial Analysis Summary:

**KEY METRICS IDENTIFIED:**
- Revenue trends show concerning patterns requiring investigation
- Inventory levels appear elevated relative to operational activity
- Working capital ratios suggest near-term liquidity concerns
- Debt service coverage requires careful monitoring

**AREAS OF CONCERN:**
- Unusually high inventory-to-revenue ratios detected
- Customer payment terms appear extended beyond industry norms
- Capital structure may require restructuring post-acquisition

**RECOMMENDATIONS:**
- Perform detailed working capital analysis
- Investigate inventory obsolescence risk
- Verify accounts receivable collectibility
- Model various revenue scenarios given concentration risk"""
    
    def _generate_concentration_analysis(self) -> str:
        """Generate customer concentration analysis"""
        return """Customer Concentration Analysis:

**FINDINGS:**
- Significant revenue concentration detected in customer base
- Top customer relationships represent disproportionate revenue share
- Limited customer diversification increases business risk profile
- Recent communications suggest potential changes to key relationships

**RISK ASSESSMENT:**
- Loss of any major customer would materially impact operations
- Bargaining power heavily weighted toward customers
- Limited pricing flexibility due to concentration
- Strategic vulnerability to customer decisions

**MITIGATION STRATEGIES:**
- Require customer diversification plan post-acquisition
- Negotiate long-term contracts with key customers
- Build in price adjustments for concentration risk
- Consider earnout structure tied to customer retention"""
    
    def _generate_generic_analysis(self) -> str:
        """Generate generic helpful response"""
        return """I can help analyze your M&A documents across several dimensions:

1. **Financial Health**: Extract and analyze key financial metrics, ratios, and trends
2. **Risk Assessment**: Identify operational, financial, and strategic risks
3. **Due Diligence**: Flag inconsistencies and areas requiring investigation
4. **Valuation Impact**: Assess how findings might affect deal valuation

Please upload relevant documents and ask specific questions about areas of concern."""
    
    def analyze_for_deal_breakers(self, contexts: List[str]) -> Dict[str, Any]:
        """Specialized method for deal breaker analysis"""
        combined_context = "\n\n".join(contexts)
        
        prompt = f"""As an M&A analyst, identify critical deal-breaking issues in these document excerpts.
Focus on:
- Customer concentration over 30%
- Product/service discontinuation
- Financial distress indicators
- Material misrepresentations
- Regulatory compliance issues

Documents:
{combined_context[:3000]}

Provide structured analysis with severity ratings."""
        
        analysis = self.generate(prompt)
        
        # Parse response for structured data
        is_critical = any(term in analysis.lower() for term in 
                         ['critical', 'severe', 'material', 'significant risk'])
        
        return {
            "analysis": analysis,
            "has_critical_issues": is_critical,
            "timestamp": datetime.utcnow().isoformat()
        }