# core/entity_extractor.py
"""
Entity Extraction for Knowledge Graph
Extracts organizations, products, financial metrics, people, dates, and more
from M&A documents using NLP and pattern matching
"""

import re
import spacy
from typing import List, Dict, Any, Set, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import logging

from graph.schema import (
    NodeLabel, 
    OrganizationType, 
    ProductStatus, 
    EventType
)

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Production-grade entity extractor for M&A documents
    Extracts entities dynamically from any document text
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize entity extractor with NLP models
        
        Args:
            spacy_model: spaCy model name to use
        """
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            logger.error(f"spaCy model '{spacy_model}' not found. Run: python -m spacy download {spacy_model}")
            raise
        
        # Initialize pattern matchers
        self._init_patterns()
        
        # Cache for entity deduplication
        self._entity_cache = defaultdict(set)
    
    def _init_patterns(self):
        """Initialize regex patterns for entity extraction"""
        
        # Money patterns (supports $, €, £, ¥, etc.)
        self.money_pattern = re.compile(
            r'(?:USD|EUR|GBP|JPY|CNY|[$€£¥])\s*'
            r'([\d,]+(?:\.\d{1,2})?)\s*'
            r'(?:(million|billion|thousand|[MBK]))?',
            re.IGNORECASE
        )
        
        # Alternative money pattern (number first)
        self.money_pattern_alt = re.compile(
            r'([\d,]+(?:\.\d{1,2})?)\s*'
            r'(?:(million|billion|thousand|[MBK]))?\s*'
            r'(?:USD|EUR|GBP|JPY|CNY|dollars?|euros?|pounds?|yen)',
            re.IGNORECASE
        )
        
        # Percentage pattern
        self.percentage_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*%')
        
        # Date patterns
        self.date_patterns = [
            re.compile(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})', re.IGNORECASE),
            re.compile(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})'),
            re.compile(r'(Q[1-4])\s+(\d{4})', re.IGNORECASE),
            re.compile(r'(\d{4})', re.IGNORECASE)  # Just year
        ]
        
        # Email pattern
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Product/service indicators
        self.product_indicators = [
            'product', 'service', 'solution', 'platform', 'system',
            'technology', 'software', 'hardware', 'device', 'tool'
        ]
        
        # Financial metric keywords
        self.financial_keywords = {
            'revenue': ['revenue', 'sales', 'turnover', 'gross receipts', 'income'],
            'profit': ['profit', 'earnings', 'net income', 'ebitda', 'operating income'],
            'assets': ['assets', 'total assets', 'current assets', 'fixed assets'],
            'liabilities': ['liabilities', 'debt', 'obligations', 'payables'],
            'equity': ['equity', 'shareholders equity', 'net worth'],
            'inventory': ['inventory', 'stock', 'inventories'],
            'cash': ['cash', 'cash flow', 'liquidity']
        }
        
        # Business relationship keywords
        self.relationship_keywords = {
            'customer': ['customer', 'client', 'buyer', 'purchaser'],
            'supplier': ['supplier', 'vendor', 'provider'],
            'competitor': ['competitor', 'rival', 'competition'],
            'partner': ['partner', 'alliance', 'collaboration']
        }
        
        # Event keywords
        self.event_keywords = {
            'directive': ['directive', 'order', 'instruction', 'mandate', 'requirement'],
            'filing': ['filing', 'report', '10-K', '10-Q', '8-K', 'disclosure'],
            'transaction': ['acquisition', 'merger', 'purchase', 'sale', 'deal'],
            'discontinue': ['discontinue', 'stop', 'cease', 'end', 'terminate', 'halt']
        }
    
    def extract_entities(self, text: str, source_doc: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract all entities from text
        
        Args:
            text: Document text to analyze
            source_doc: Source document filename
            
        Returns:
            Dictionary of extracted entities by type
        """
        if not text or not text.strip():
            return self._empty_result()
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract different entity types
        entities = {
            'organizations': self._extract_organizations(doc, text),
            'products': self._extract_products(doc, text),
            'financial_metrics': self._extract_financial_metrics(doc, text),
            'people': self._extract_people(doc),
            'dates': self._extract_dates(text),
            'events': self._extract_events(doc, text),
            'locations': self._extract_locations(doc),
            'percentages': self._extract_percentages(text),
            'emails': self._extract_emails(text)
        }
        
        # Add source document to all entities
        if source_doc:
            for entity_type in entities:
                for entity in entities[entity_type]:
                    entity['source_doc'] = source_doc
        
        # Log extraction summary
        total = sum(len(entities[t]) for t in entities)
        logger.info(f"Extracted {total} entities from text ({len(text)} chars)")
        
        return entities
    
    def _extract_organizations(self, doc, text: str) -> List[Dict[str, Any]]:
        """Extract organization names using NER and patterns"""
        organizations = []
        seen = set()
        
        # Extract using spaCy NER
        for ent in doc.ents:
            if ent.label_ == "ORG":
                name = ent.text.strip()
                
                # Clean up common issues
                name = re.sub(r'\s+', ' ', name)
                name = name.strip('.,;:')
                
                if len(name) > 2 and name.lower() not in seen:
                    seen.add(name.lower())
                    
                    # Determine organization type from context
                    org_type = self._determine_org_type(name, text)
                    
                    organizations.append({
                        'name': name,
                        'type': org_type,
                        'confidence': 0.85,
                        'extraction_method': 'spacy_ner'
                    })
        
        return organizations
    
    def _determine_org_type(self, org_name: str, text: str) -> str:
        """Determine organization type from context"""
        # Find context around organization name
        pattern = re.compile(rf'.{{0,100}}{re.escape(org_name)}.{{0,100}}', re.IGNORECASE)
        matches = pattern.findall(text)
        
        if not matches:
            return OrganizationType.TARGET.value
        
        context = ' '.join(matches).lower()
        
        # Check for relationship keywords
        if any(kw in context for kw in self.relationship_keywords['customer']):
            return OrganizationType.CUSTOMER.value
        elif any(kw in context for kw in self.relationship_keywords['supplier']):
            return OrganizationType.SUPPLIER.value
        elif any(kw in context for kw in self.relationship_keywords['competitor']):
            return OrganizationType.COMPETITOR.value
        elif any(kw in context for kw in self.relationship_keywords['partner']):
            return OrganizationType.PARTNER.value
        elif 'acquir' in context or 'target' in context:
            return OrganizationType.TARGET.value
        elif 'buyer' in context or 'purchasing' in context:
            return OrganizationType.ACQUIRER.value
        
        return OrganizationType.TARGET.value
    
    def _extract_products(self, doc, text: str) -> List[Dict[str, Any]]:
        """Extract product/service names"""
        products = []
        seen = set()
        
        # Look for noun phrases that might be products
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            
            # Check if chunk mentions product indicators
            if any(indicator in chunk_text for indicator in self.product_indicators):
                # Extract the actual product name (remove indicators)
                product_name = chunk.text.strip()
                
                # Clean up
                for indicator in self.product_indicators:
                    product_name = re.sub(rf'\b{indicator}\b', '', product_name, flags=re.IGNORECASE)
                
                product_name = product_name.strip()
                
                if len(product_name) > 3 and product_name.lower() not in seen:
                    seen.add(product_name.lower())
                    
                    # Determine product status from context
                    status = self._determine_product_status(product_name, text)
                    
                    products.append({
                        'name': product_name,
                        'status': status,
                        'confidence': 0.70,
                        'extraction_method': 'noun_phrase'
                    })
        
        # Also look for specific product patterns (capitalized multi-word terms)
        product_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b')
        for match in product_pattern.finditer(text):
            product_name = match.group(1)
            
            if product_name.lower() not in seen and len(product_name) > 5:
                # Avoid common false positives
                if not any(word in product_name for word in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']):
                    seen.add(product_name.lower())
                    
                    status = self._determine_product_status(product_name, text)
                    
                    products.append({
                        'name': product_name,
                        'status': status,
                        'confidence': 0.60,
                        'extraction_method': 'pattern'
                    })
        
        return products[:20]  # Limit to top 20 most confident
    
    def _determine_product_status(self, product_name: str, text: str) -> str:
        """Determine product status from context"""
        pattern = re.compile(rf'.{{0,150}}{re.escape(product_name)}.{{0,150}}', re.IGNORECASE)
        matches = pattern.findall(text)
        
        if not matches:
            return ProductStatus.ACTIVE.value
        
        context = ' '.join(matches).lower()
        
        # Check for status indicators
        if any(word in context for word in ['discontinue', 'discontinued', 'stop', 'cease', 'end']):
            return ProductStatus.DISCONTINUED.value
        elif any(word in context for word in ['growing', 'increasing', 'expansion', 'strong']):
            return ProductStatus.GROWING.value
        elif any(word in context for word in ['declining', 'decreasing', 'weak', 'down']):
            return ProductStatus.DECLINING.value
        
        return ProductStatus.ACTIVE.value
    
    def _extract_financial_metrics(self, doc, text: str) -> List[Dict[str, Any]]:
        """Extract financial metrics (money, percentages, ratios)"""
        metrics = []
        
        # Extract money values
        for pattern in [self.money_pattern, self.money_pattern_alt]:
            for match in pattern.finditer(text):
                try:
                    value_str = match.group(1).replace(',', '')
                    value = float(value_str)
                    
                    # Check for multiplier
                    multiplier = 1
                    if len(match.groups()) > 1 and match.group(2):
                        mult_str = match.group(2).lower()
                        if mult_str in ['m', 'million']:
                            multiplier = 1_000_000
                        elif mult_str in ['b', 'billion']:
                            multiplier = 1_000_000_000
                        elif mult_str in ['k', 'thousand']:
                            multiplier = 1_000
                    
                    final_value = value * multiplier
                    
                    # Get context to determine metric type
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end].lower()
                    
                    # Determine metric name
                    metric_name = self._determine_metric_name(context)
                    
                    metrics.append({
                        'name': metric_name,
                        'value': final_value,
                        'unit': 'USD',
                        'original_text': match.group(0),
                        'context': context[:200],
                        'confidence': 0.90
                    })
                    
                except (ValueError, IndexError):
                    continue
        
        return metrics
    
    def _determine_metric_name(self, context: str) -> str:
        """Determine financial metric name from context"""
        for metric_type, keywords in self.financial_keywords.items():
            if any(kw in context for kw in keywords):
                return metric_type
        
        return "financial_metric"
    
    def _extract_people(self, doc) -> List[Dict[str, Any]]:
        """Extract person names using NER"""
        people = []
        seen = set()
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                
                # Basic validation
                if len(name) > 3 and name.lower() not in seen:
                    seen.add(name.lower())
                    
                    people.append({
                        'name': name,
                        'confidence': 0.85,
                        'extraction_method': 'spacy_ner'
                    })
        
        return people
    
    def _extract_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract dates using patterns"""
        dates = []
        seen = set()
        
        for pattern in self.date_patterns:
            for match in pattern.finditer(text):
                date_str = match.group(0)
                
                if date_str.lower() not in seen:
                    seen.add(date_str.lower())
                    
                    # Try to parse to standard format
                    parsed_date = self._parse_date(date_str)
                    
                    dates.append({
                        'original': date_str,
                        'parsed': parsed_date,
                        'confidence': 0.90
                    })
        
        return dates
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse date string to ISO format"""
        try:
            # Try various date formats
            for fmt in ['%B %d, %Y', '%m/%d/%Y', '%d/%m/%Y', '%Y']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            
            # Handle quarter format (Q1 2024)
            if date_str.startswith('Q'):
                quarter_match = re.match(r'Q([1-4])\s+(\d{4})', date_str, re.IGNORECASE)
                if quarter_match:
                    quarter = int(quarter_match.group(1))
                    year = int(quarter_match.group(2))
                    month = (quarter - 1) * 3 + 1
                    return f"{year}-{month:02d}-01"
        
        except Exception:
            pass
        
        return None
    
    def _extract_events(self, doc, text: str) -> List[Dict[str, Any]]:
        """Extract events (directives, filings, transactions)"""
        events = []
        
        for event_type, keywords in self.event_keywords.items():
            for keyword in keywords:
                pattern = re.compile(rf'.{{0,100}}\b{keyword}\b.{{0,100}}', re.IGNORECASE)
                
                for match in pattern.finditer(text):
                    context = match.group(0)
                    
                    events.append({
                        'type': event_type,
                        'description': context.strip(),
                        'confidence': 0.75,
                        'extraction_method': 'keyword'
                    })
        
        return events[:10]  # Limit to top 10
    
    def _extract_locations(self, doc) -> List[Dict[str, Any]]:
        """Extract location entities"""
        locations = []
        seen = set()
        
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:  # Geo-political entity or location
                name = ent.text.strip()
                
                if len(name) > 2 and name.lower() not in seen:
                    seen.add(name.lower())
                    
                    locations.append({
                        'name': name,
                        'type': 'city' if ent.label_ == "GPE" else 'location',
                        'confidence': 0.80
                    })
        
        return locations
    
    def _extract_percentages(self, text: str) -> List[Dict[str, Any]]:
        """Extract percentage values"""
        percentages = []
        
        for match in self.percentage_pattern.finditer(text):
            try:
                value = float(match.group(1))
                
                # Get context
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]
                
                percentages.append({
                    'value': value,
                    'original_text': match.group(0),
                    'context': context,
                    'confidence': 0.95
                })
            except ValueError:
                continue
        
        return percentages
    
    def _extract_emails(self, text: str) -> List[Dict[str, Any]]:
        """Extract email addresses"""
        emails = []
        seen = set()
        
        for match in self.email_pattern.finditer(text):
            email = match.group(0)
            
            if email.lower() not in seen:
                seen.add(email.lower())
                
                emails.append({
                    'email': email,
                    'confidence': 0.98
                })
        
        return emails
    
    def _empty_result(self) -> Dict[str, List]:
        """Return empty result structure"""
        return {
            'organizations': [],
            'products': [],
            'financial_metrics': [],
            'people': [],
            'dates': [],
            'events': [],
            'locations': [],
            'percentages': [],
            'emails': []
        }
    
    def extract_and_deduplicate(
        self, 
        texts: List[str], 
        source_docs: List[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract entities from multiple texts and deduplicate
        
        Args:
            texts: List of document texts
            source_docs: List of source document names
            
        Returns:
            Deduplicated entities
        """
        all_entities = self._empty_result()
        
        for i, text in enumerate(texts):
            source = source_docs[i] if source_docs and i < len(source_docs) else None
            entities = self.extract_entities(text, source)
            
            # Merge with existing
            for entity_type in all_entities:
                all_entities[entity_type].extend(entities[entity_type])
        
        # Deduplicate
        deduplicated = self._deduplicate_entities(all_entities)
        
        return deduplicated
    
    def _deduplicate_entities(self, entities: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Deduplicate entities by name/value"""
        deduplicated = self._empty_result()
        
        for entity_type in entities:
            seen = set()
            
            for entity in entities[entity_type]:
                # Create unique key based on entity type
                if entity_type == 'organizations':
                    key = entity['name'].lower()
                elif entity_type == 'products':
                    key = entity['name'].lower()
                elif entity_type == 'people':
                    key = entity['name'].lower()
                elif entity_type == 'financial_metrics':
                    key = f"{entity['name']}_{entity['value']}"
                elif entity_type == 'dates':
                    key = entity['original'].lower()
                elif entity_type == 'emails':
                    key = entity['email'].lower()
                else:
                    key = str(entity)
                
                if key not in seen:
                    seen.add(key)
                    deduplicated[entity_type].append(entity)
        
        return deduplicated


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_entities_from_text(text: str, source_doc: str = None) -> Dict[str, List[Dict]]:
    """
    Convenience function for quick entity extraction
    
    Args:
        text: Text to extract from
        source_doc: Source document name
        
    Returns:
        Extracted entities
    """
    extractor = EntityExtractor()
    return extractor.extract_entities(text, source_doc)


if __name__ == "__main__":
    # Test the entity extractor
    test_text = """
    CFC Corporation sold $4.26 million of holographic stripe products to Visa Inc. 
    in March 2006. However, a directive from Visa on March 15, 2006 required CFC 
    to discontinue production. This represents 100% of CFC's revenue from this 
    product line. The company had $346,000 in inventory at the time.
    
    Contact: john.doe@cfc.com for more information.
    """
    
    print("="*80)
    print("ENTITY EXTRACTOR TEST")
    print("="*80)
    
    extractor = EntityExtractor()
    entities = extractor.extract_entities(test_text, "test_document.pdf")
    
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"\n{entity_type.upper()} ({len(entity_list)}):")
            for entity in entity_list[:5]:  # Show first 5
                print(f"  {entity}")