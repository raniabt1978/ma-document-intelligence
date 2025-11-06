import re
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import difflib

@dataclass
class Inconsistency:
    fact1: str
    fact2: str
    source1: str
    source2: str
    date1: str
    date2: str
    inconsistency_type: str
    severity: str
    explanation: str

class InconsistencyDetector:
    def __init__(self):
        # Types of facts to extract and compare
        self.fact_patterns = {
            "financial": [
                r'(?:revenue|sales).*?\$[\d,]+\.?\d*[MBK]?',
                r'inventory.*?\$[\d,]+\.?\d*[MBK]?',
                r'(?:total\s+)?assets.*?\$[\d,]+\.?\d*[MBK]?',
                r'(?:debt|liabilities).*?\$[\d,]+\.?\d*[MBK]?',
                r'\$[\d,]+\.?\d*[MBK]?\s*(?:million|billion|thousand)?\s+(?:in\s+)?(?:revenue|sales|inventory)'
            ],
            "product_status": [
                r'product.*?(?:discontinued|growing|declining|stable|strong)',
                r'(?:will|to|must)\s+(?:stop|cease|discontinue|end)\s+(?:production|manufacturing|product)',
                r'(?:growing|increasing|expanding|strong)\s+(?:product|business|sales|demand)',
                r'directive.*?(?:stop|cease|discontinue)',
                r'(?:stop|cease|discontinue)\s+(?:production|manufacturing).*?(?:of|for)\s+\w+',
                r'holographic\s+stripe.*?(?:production|manufacturing|product)',
                r'Visa.*?(?:directive|instruction|order|requirement)'
            ],
            "percentage": [
                r'(\d+)%\s+of\s+(?:revenue|sales|business|customers)',
                r'(?:represents?|accounts?\s+for|comprises?)\s+(\d+)%',
                r'(?:all|entire|whole)\s+(?:revenue|sales|business)',
                r'(?:sole|only|single|exclusive)\s+(?:customer|client|supplier)',
                r'100%\s+(?:of\s+)?(?:revenue|sales|business)'
            ],
            "customer_dependency": [
                r'(?:Visa|Mastercard|customer).*?(?:all|entire|100%|sole|only)',
                r'(?:single|one)\s+customer.*?(?:revenue|sales)',
                r'customer\s+concentration',
                r'(?:dependent|reliant|reliance)\s+(?:on|upon)\s+(?:Visa|single|one)'
            ],
            "dates": [
                r'(?:by|before|after|since|until|in)\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
                r'(?:Q[1-4]|quarter)\s+\d{4}',
                r'\d{4}\s+(?:annual|year|fiscal)'
            ],
            "quantities": [
                r'\d+\s+(?:units?|items?|products?)',
                r'(?:produced?|manufactured?|sold?)\s+\d+',
                r'production\s+(?:of\s+)?\d+'
            ]
        }
        
    def find_inconsistencies(self, documents: List[Dict[str, Any]]) -> List[Inconsistency]:
        """Find inconsistencies across multiple documents"""
        inconsistencies = []
        
        # Extract facts from each document
        all_facts = []
        for doc in documents:
            facts = self._extract_facts(
                doc['text'], 
                doc.get('filename', 'Unknown'),
                doc.get('date', 'Unknown')
            )
            all_facts.extend(facts)
        
        # Compare facts between documents
        for i in range(len(all_facts)):
            for j in range(i + 1, len(all_facts)):
                fact1, fact2 = all_facts[i], all_facts[j]
                
                # Skip if from same document
                if fact1['source'] == fact2['source']:
                    continue
                
                # Check for contradictions
                inconsistency = self._check_contradiction(fact1, fact2)
                if inconsistency:
                    inconsistencies.append(inconsistency)
        
        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        inconsistencies.sort(key=lambda x: severity_order.get(x.severity, 4))
        
        return inconsistencies
    
    def _extract_facts(self, text: str, source: str, date: str) -> List[Dict[str, Any]]:
        """Extract facts from document text"""
        facts = []
        
        # Split into sentences
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check each fact pattern type
            for fact_type, patterns in self.fact_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, sentence, re.IGNORECASE)
                    for match in matches:
                        # Extract context around the match
                        start = max(0, match.start() - 50)
                        end = min(len(sentence), match.end() + 50)
                        context = sentence[start:end]
                        
                        fact = {
                            'type': fact_type,
                            'text': match.group(),
                            'context': context,
                            'full_sentence': sentence,
                            'source': source,
                            'date': date,
                            'value': self._extract_value(match.group(), fact_type)
                        }
                        facts.append(fact)
        
        return facts
    
    def _check_contradiction(self, fact1: Dict, fact2: Dict) -> Inconsistency:
        """Check if two facts contradict each other"""
        
        # Type 1: Financial value contradictions (only for same metric)
        if fact1['type'] == 'financial' and fact2['type'] == 'financial':
            # Check if discussing same metric
            if self._same_financial_metric(fact1['text'], fact2['text']):
                val1, val2 = fact1['value'], fact2['value']
                if val1 and val2 and abs(val1 - val2) / max(val1, val2) > 0.1:  # >10% difference
                    return Inconsistency(
                        fact1=fact1['text'],
                        fact2=fact2['text'],
                        source1=fact1['source'],
                        source2=fact2['source'],
                        date1=fact1['date'],
                        date2=fact2['date'],
                        inconsistency_type="financial_mismatch",
                        severity="high" if abs(val1 - val2) / max(val1, val2) > 0.5 else "medium",
                        explanation=f"Financial values differ by {abs(val1-val2)/max(val1,val2)*100:.1f}%"
                    )
        
        # Type 2: Product status contradictions (CRITICAL - like CFC case)
        if fact1['type'] == 'product_status' and fact2['type'] == 'product_status':
            status1 = self._extract_product_status(fact1['full_sentence'])
            status2 = self._extract_product_status(fact2['full_sentence'])
            
            if status1 and status2 and status1 != status2:
                # Check for critical contradictions
                if ('stop' in status1 and 'growing' in status2) or ('stop' in status2 and 'growing' in status1):
                    return Inconsistency(
                        fact1=fact1['full_sentence'],
                        fact2=fact2['full_sentence'],
                        source1=fact1['source'],
                        source2=fact2['source'],
                        date1=fact1['date'],
                        date2=fact2['date'],
                        inconsistency_type="product_status_contradiction",
                        severity="critical",
                        explanation="CRITICAL: Document claims product is growing while another indicates production must stop"
                    )
                elif any(neg in [status1, status2] for neg in ['stopping', 'declining']) and any(pos in [status1, status2] for pos in ['growing', 'strong']):
                    return Inconsistency(
                        fact1=fact1['context'],
                        fact2=fact2['context'],
                        source1=fact1['source'],
                        source2=fact2['source'],
                        date1=fact1['date'],
                        date2=fact2['date'],
                        inconsistency_type="status_mismatch",
                        severity="high",
                        explanation=f"Product status contradiction: '{status1}' vs '{status2}'"
                    )
        
        # Type 3: Customer dependency (100% concentration is critical)
        if fact1['type'] == 'customer_dependency' or fact2['type'] == 'customer_dependency':
            # Check if one mentions 100% or sole customer
            if any(term in str(fact1.get('full_sentence', '')).lower() for term in ['100%', 'sole', 'only', 'all revenue']):
                return Inconsistency(
                    fact1=fact1['full_sentence'],
                    fact2="Business concentration risk identified",
                    source1=fact1['source'],
                    source2=fact1['source'],
                    date1=fact1['date'],
                    date2=fact1['date'],
                    inconsistency_type="concentration_risk",
                    severity="critical",
                    explanation="CRITICAL: 100% customer concentration detected - extreme business risk"
                )
        
        # Type 4: Percentage contradictions
        if fact1['type'] == 'percentage' and fact2['type'] == 'percentage':
            pct1, pct2 = fact1['value'], fact2['value']
            if pct1 and pct2 and abs(pct1 - pct2) > 10:  # >10 percentage point difference
                return Inconsistency(
                    fact1=fact1['text'],
                    fact2=fact2['text'],
                    source1=fact1['source'],
                    source2=fact2['source'],
                    date1=fact1['date'],
                    date2=fact2['date'],
                    inconsistency_type="percentage_mismatch",
                    severity="medium",
                    explanation=f"Percentage values differ: {pct1}% vs {pct2}%"
                )
        
        # Type 5: Cross-type contradictions (e.g., growing sales vs stop directive)
        if (fact1['type'] == 'financial' and 'growing' in fact1.get('full_sentence', '').lower() and 
            fact2['type'] == 'product_status' and 'stop' in fact2.get('full_sentence', '').lower()):
            return Inconsistency(
                fact1=fact1['full_sentence'],
                fact2=fact2['full_sentence'],
                source1=fact1['source'],
                source2=fact2['source'],
                date1=fact1['date'],
                date2=fact2['date'],
                inconsistency_type="business_contradiction",
                severity="critical",
                explanation="CRITICAL: Financial growth claimed while production cessation ordered"
            )
        
        return None
    
    def _extract_value(self, text: str, fact_type: str) -> Any:
        """Extract numerical value from fact text"""
        if fact_type == 'financial':
            # Extract dollar amount
            money_match = re.search(r'\$?([\d,]+\.?\d*)([MBK])?', text)
            if money_match:
                try:
                    # Get the numeric part and remove commas
                    num_str = money_match.group(1).replace(',', '')
                    if num_str:  # Check if not empty
                        value = float(num_str)
                        # Check for multiplier
                        if money_match.group(2):
                            if money_match.group(2).upper() == 'M':
                                value *= 1_000_000
                            elif money_match.group(2).upper() == 'B':
                                value *= 1_000_000_000
                            elif money_match.group(2).upper() == 'K':
                                value *= 1_000
                        return value
                except ValueError:
                    return None
        
        elif fact_type == 'percentage':
            pct_match = re.search(r'(\d+)%', text)
            if pct_match:
                try:
                    return int(pct_match.group(1))
                except ValueError:
                    return None
        
        elif fact_type == 'quantities':
            qty_match = re.search(r'(\d+)', text)
            if qty_match:
                try:
                    return int(qty_match.group(1))
                except ValueError:
                    return None
        
        return None
    
    def _same_financial_metric(self, text1: str, text2: str) -> bool:
        """Check if two financial facts discuss the same metric"""
        metric_keywords = ['revenue', 'sales', 'inventory', 'assets', 'debt', 'liabilities']
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        for keyword in metric_keywords:
            if keyword in text1_lower and keyword in text2_lower:
                return True
        
        return False
    
    def _extract_product_status(self, text: str) -> str:
        """Extract product status from text"""
        text_lower = text.lower()
        
        # Look for specific patterns in the full sentence
        if any(phrase in text_lower for phrase in ['stop production', 'cease production', 'discontinue production', 'end production']):
            return 'stopping'
        elif any(phrase in text_lower for phrase in ['must stop', 'will stop', 'to stop', 'directive to stop']):
            return 'stopping'
        elif any(word in text_lower for word in ['growing', 'increasing', 'expanding', 'strong demand']):
            return 'growing'
        elif 'declining' in text_lower:
            return 'declining'
        elif 'stable' in text_lower:
            return 'stable'
        
        return None
    
    def generate_inconsistency_report(self, inconsistencies: List[Inconsistency]) -> Dict[str, Any]:
        """Generate summary report of inconsistencies"""
        if not inconsistencies:
            return {
                "summary": "No inconsistencies detected",
                "critical_count": 0,
                "total_count": 0,
                "recommendations": []
            }
        
        critical = [i for i in inconsistencies if i.severity == "critical"]
        high = [i for i in inconsistencies if i.severity == "high"]
        
        recommendations = []
        if critical:
            recommendations.append("URGENT: Critical inconsistencies found - immediate review required")
            recommendations.append("Recommend halting deal proceedings until resolved")
        if high:
            recommendations.append("High-priority inconsistencies require management clarification")
        
        return {
            "summary": f"Found {len(inconsistencies)} inconsistencies ({len(critical)} critical)",
            "critical_count": len(critical),
            "high_count": len(high),
            "total_count": len(inconsistencies),
            "most_severe": inconsistencies[0] if inconsistencies else None,
            "recommendations": recommendations,
            "by_type": self._group_by_type(inconsistencies)
        }
    
    def _group_by_type(self, inconsistencies: List[Inconsistency]) -> Dict[str, int]:
        """Group inconsistencies by type"""
        groups = {}
        for inc in inconsistencies:
            groups[inc.inconsistency_type] = groups.get(inc.inconsistency_type, 0) + 1
        return groups