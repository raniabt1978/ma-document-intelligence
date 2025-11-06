# core/relationship_mapper.py
"""
Relationship Mapper for Knowledge Graph
Maps relationships between extracted entities to build graph connections
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import logging

from graph.schema import RelationshipType, Severity

logger = logging.getLogger(__name__)


class RelationshipMapper:
    """
    Maps relationships between entities for knowledge graph construction
    Finds connections like SELLS_TO, DEPENDS_ON, CONTRADICTS, etc.
    """
    
    def __init__(self):
        """Initialize relationship mapper with patterns"""
        self._init_relationship_patterns()
    
    def _init_relationship_patterns(self):
        """Initialize regex patterns for relationship detection"""
        
        # Business relationship patterns
        self.sells_patterns = [
            r'(\w+)\s+(?:sold|sells|selling|sell)\s+(?:to|for)\s+(\w+)',
            r'(\w+)\s+(?:revenue|sales)\s+(?:from|to)\s+(\w+)',
            r'(\w+)\s+(?:customer|client|buyer)\s+(?:is|was|include[s]?)\s+(\w+)'
        ]
        
        self.produces_patterns = [
            r'(\w+)\s+(?:produces|produce|manufactured|makes?)\s+(.+?)(?:\.|,|$)',
            r'(\w+)\s+(?:product|service)\s+(?:is|include[s]?)\s+(.+?)(?:\.|,|$)',
            r'(.+?)\s+(?:produced|manufactured)\s+by\s+(\w+)'
        ]
        
        self.acquiring_patterns = [
            r'(\w+)\s+(?:acquir(?:ing|ed)|purchas(?:ing|ed)|bought?)\s+(\w+)',
            r'(\w+)\s+(?:acquisition|purchase|deal)\s+(?:of|with)\s+(\w+)',
            r'(\w+)\s+to\s+(?:acquire|purchase|buy)\s+(\w+)'
        ]
        
        # Dependency patterns
        self.depends_patterns = [
            r'(\d+)%\s+(?:of|from)\s+(?:revenue|sales)\s+(?:from|to)\s+(\w+)',
            r'(?:sole|only|single|entire|all)\s+(?:customer|supplier)\s+(?:is|was)\s+(\w+)',
            r'(\w+)\s+(?:reliance|dependence|dependency)\s+on\s+(\w+)',
            r'100%\s+(?:of|from)\s+(.+?)\s+(?:from|to)\s+(\w+)'
        ]
        
        # Event patterns
        self.directive_patterns = [
            r'(\w+)\s+(?:directive|order|instruction)\s+to\s+(.+?)(?:\.|,)',
            r'(\w+)\s+(?:required|mandated|ordered)\s+(\w+)\s+to\s+(.+?)(?:\.|,)',
            r'directive\s+from\s+(\w+)'
        ]
        
        self.discontinue_patterns = [
            r'(?:discontinue|stop|cease|end|halt|terminate)\s+(?:production|manufacturing)\s+(?:of\s+)?(.+?)(?:\.|,)',
            r'(.+?)\s+(?:was|is|been)\s+(?:discontinued|stopped|ceased)'
        ]
        
        # Contradiction patterns
        self.contradiction_indicators = [
            ('growing', 'discontinued'),
            ('increasing', 'stop'),
            ('strong', 'cease'),
            ('expansion', 'terminate'),
            ('positive', 'negative')
        ]
    
    def map_relationships(
        self, 
        entities: Dict[str, List[Dict[str, Any]]],
        text: str
    ) -> List[Dict[str, Any]]:
        """
        Map relationships between entities
        
        Args:
            entities: Extracted entities from entity_extractor
            text: Original document text for context
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        # Extract different relationship types
        relationships.extend(self._find_business_relationships(entities, text))
        relationships.extend(self._find_financial_relationships(entities, text))
        relationships.extend(self._find_dependency_relationships(entities, text))
        relationships.extend(self._find_event_relationships(entities, text))
        relationships.extend(self._find_contradictions(entities, text))
        relationships.extend(self._find_temporal_relationships(entities, text))
        
        # Deduplicate
        relationships = self._deduplicate_relationships(relationships)
        
        logger.info(f"Mapped {len(relationships)} relationships")
        
        return relationships
    
    def _find_business_relationships(
        self, 
        entities: Dict[str, List[Dict]], 
        text: str
    ) -> List[Dict]:
        """Find business relationships like SELLS_TO, PRODUCES"""
        relationships = []
        
        orgs = entities.get('organizations', [])
        products = entities.get('products', [])
        
        # SELLS_TO relationships
        for pattern in self.sells_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                seller = match.group(1)
                buyer = match.group(2)
                
                # Verify entities exist
                if self._entity_exists(seller, orgs) and self._entity_exists(buyer, orgs):
                    relationships.append({
                        'type': RelationshipType.SELLS_TO.value,
                        'from_entity': seller,
                        'from_type': 'Organization',
                        'to_entity': buyer,
                        'to_type': 'Organization',
                        'properties': {
                            'confidence': 0.85,
                            'context': match.group(0)
                        }
                    })
        
        # PRODUCES relationships
        for pattern in self.produces_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                org = match.group(1)
                product = match.group(2).strip()
                
                if self._entity_exists(org, orgs) and self._entity_exists(product, products):
                    relationships.append({
                        'type': RelationshipType.PRODUCES.value,
                        'from_entity': org,
                        'from_type': 'Organization',
                        'to_entity': product,
                        'to_type': 'Product',
                        'properties': {
                            'confidence': 0.80,
                            'context': match.group(0)
                        }
                    })
        
        # ACQUIRING relationships
        for pattern in self.acquiring_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                acquirer = match.group(1)
                target = match.group(2)
                
                if self._entity_exists(acquirer, orgs) and self._entity_exists(target, orgs):
                    relationships.append({
                        'type': RelationshipType.ACQUIRING.value,
                        'from_entity': acquirer,
                        'from_type': 'Organization',
                        'to_entity': target,
                        'to_type': 'Organization',
                        'properties': {
                            'confidence': 0.90,
                            'context': match.group(0)
                        }
                    })
        
        return relationships
    
    def _find_financial_relationships(
        self,
        entities: Dict[str, List[Dict]],
        text: str
    ) -> List[Dict]:
        """Find financial relationships like HAS_METRIC, GENERATES_REVENUE"""
        relationships = []
        
        orgs = entities.get('organizations', [])
        products = entities.get('products', [])
        metrics = entities.get('financial_metrics', [])
        
        # Link metrics to organizations/products
        for metric in metrics:
            context = metric.get('context', '')
            
            # Find which entity this metric belongs to
            for org in orgs:
                if org['name'].lower() in context.lower():
                    relationships.append({
                        'type': RelationshipType.HAS_METRIC.value,
                        'from_entity': org['name'],
                        'from_type': 'Organization',
                        'to_entity': f"{metric['name']}_{metric['value']}",
                        'to_type': 'FinancialMetric',
                        'properties': {
                            'metric_name': metric['name'],
                            'value': metric['value'],
                            'confidence': 0.85
                        }
                    })
            
            # Check if metric is from a product
            for product in products:
                if product['name'].lower() in context.lower():
                    relationships.append({
                        'type': RelationshipType.GENERATES_REVENUE.value,
                        'from_entity': product['name'],
                        'from_type': 'Product',
                        'to_entity': f"{metric['name']}_{metric['value']}",
                        'to_type': 'FinancialMetric',
                        'properties': {
                            'value': metric['value'],
                            'confidence': 0.80
                        }
                    })
        
        return relationships
    
    def _find_dependency_relationships(
        self,
        entities: Dict[str, List[Dict]],
        text: str
    ) -> List[Dict]:
        """Find DEPENDS_ON and RELIES_ON relationships (CRITICAL for M&A)"""
        relationships = []
        
        orgs = entities.get('organizations', [])
        products = entities.get('products', [])
        percentages = entities.get('percentages', [])
        
        # Look for dependency patterns
        for pattern in self.depends_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Extract percentage if present
                percentage = None
                for pct in percentages:
                    if pct['original_text'] in match.group(0):
                        percentage = pct['value']
                        break
                
                # Find entities involved
                context = match.group(0).lower()
                
                for org1 in orgs:
                    for org2 in orgs:
                        if (org1['name'].lower() in context and 
                            org2['name'].lower() in context and 
                            org1 != org2):
                            
                            # Determine criticality based on percentage
                            criticality = 'critical' if percentage and percentage >= 80 else 'high'
                            
                            relationships.append({
                                'type': RelationshipType.DEPENDS_ON.value,
                                'from_entity': org1['name'],
                                'from_type': 'Organization',
                                'to_entity': org2['name'],
                                'to_type': 'Organization',
                                'properties': {
                                    'criticality': criticality,
                                    'percentage': percentage,
                                    'confidence': 0.90,
                                    'context': match.group(0)
                                }
                            })
        
        return relationships
    
    def _find_event_relationships(
        self,
        entities: Dict[str, List[Dict]],
        text: str
    ) -> List[Dict]:
        """Find event-based relationships like ISSUES, AFFECTS"""
        relationships = []
        
        orgs = entities.get('organizations', [])
        products = entities.get('products', [])
        events = entities.get('events', [])
        
        # ISSUES relationships (organization issues directive/order)
        for pattern in self.directive_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                issuer = match.group(1) if match.lastindex >= 1 else None
                
                if issuer and self._entity_exists(issuer, orgs):
                    # Create event entity
                    event_desc = match.group(0)
                    
                    relationships.append({
                        'type': RelationshipType.ISSUES.value,
                        'from_entity': issuer,
                        'from_type': 'Organization',
                        'to_entity': event_desc[:50],  # Truncate
                        'to_type': 'Event',
                        'properties': {
                            'confidence': 0.85,
                            'event_type': 'directive'
                        }
                    })
        
        # AFFECTS relationships (event affects product/org)
        for event in events:
            event_context = event.get('description', '').lower()
            
            # Check if event affects any product
            for product in products:
                if product['name'].lower() in event_context:
                    relationships.append({
                        'type': RelationshipType.AFFECTS.value,
                        'from_entity': event['description'][:50],
                        'from_type': 'Event',
                        'to_entity': product['name'],
                        'to_type': 'Product',
                        'properties': {
                            'severity': 'critical' if 'stop' in event_context or 'cease' in event_context else 'high',
                            'confidence': 0.80
                        }
                    })
        
        return relationships
    
    def _find_contradictions(
        self,
        entities: Dict[str, List[Dict]],
        text: str
    ) -> List[Dict]:
        """Find CONTRADICTS relationships (CRITICAL for deal analysis)"""
        relationships = []
        
        products = entities.get('products', [])
        events = entities.get('events', [])
        
        # Check for contradictory product statuses
        for i, prod1 in enumerate(products):
            for prod2 in products[i+1:]:
                # Same product with different status?
                if (prod1['name'].lower() == prod2['name'].lower() and
                    prod1['status'] != prod2['status']):
                    
                    # Check if status contradiction is significant
                    status1 = prod1['status']
                    status2 = prod2['status']
                    
                    if self._is_contradictory_status(status1, status2):
                        relationships.append({
                            'type': RelationshipType.CONTRADICTS.value,
                            'from_entity': f"{prod1['name']}_status1",
                            'from_type': 'Product',
                            'to_entity': f"{prod2['name']}_status2",
                            'to_type': 'Product',
                            'properties': {
                                'severity': 'critical',
                                'explanation': f"Product status contradiction: {status1} vs {status2}",
                                'confidence': 0.90
                            }
                        })
        
        # Check for contradictory events (e.g., "growing" vs "stop production")
        for indicator1, indicator2 in self.contradiction_indicators:
            if indicator1 in text.lower() and indicator2 in text.lower():
                relationships.append({
                    'type': RelationshipType.CONTRADICTS.value,
                    'from_entity': f"statement_{indicator1}",
                    'from_type': 'Event',
                    'to_entity': f"statement_{indicator2}",
                    'to_type': 'Event',
                    'properties': {
                        'severity': 'critical',
                        'explanation': f"Contradictory statements: '{indicator1}' vs '{indicator2}'",
                        'confidence': 0.85,
                        'aspect': 'business_direction'
                    }
                })
        
        return relationships
    
    def _find_temporal_relationships(
        self,
        entities: Dict[str, List[Dict]],
        text: str
    ) -> List[Dict]:
        """Find temporal relationships like OCCURS_BEFORE, INVALIDATES"""
        relationships = []
        
        dates = entities.get('dates', [])
        events = entities.get('events', [])
        
        # Sort dates chronologically
        sorted_dates = sorted(
            [d for d in dates if d.get('parsed')],
            key=lambda x: x['parsed']
        )
        
        # Create OCCURS_BEFORE relationships
        for i in range(len(sorted_dates) - 1):
            date1 = sorted_dates[i]
            date2 = sorted_dates[i + 1]
            
            relationships.append({
                'type': RelationshipType.OCCURS_BEFORE.value,
                'from_entity': date1['original'],
                'from_type': 'Date',
                'to_entity': date2['original'],
                'to_type': 'Date',
                'properties': {
                    'confidence': 0.95
                }
            })
        
        return relationships
    
    def _is_contradictory_status(self, status1: str, status2: str) -> bool:
        """Check if two product statuses contradict each other"""
        contradictions = [
            ('growing', 'discontinued'),
            ('growing', 'declining'),
            ('active', 'discontinued'),
        ]
        
        status_pair = (status1.lower(), status2.lower())
        reverse_pair = (status2.lower(), status1.lower())
        
        return status_pair in contradictions or reverse_pair in contradictions
    
    def _entity_exists(self, name: str, entity_list: List[Dict]) -> bool:
        """Check if entity name exists in entity list"""
        name_lower = name.lower().strip()
        
        for entity in entity_list:
            entity_name = entity.get('name', '').lower().strip()
            
            # Exact match or contains
            if name_lower == entity_name or name_lower in entity_name or entity_name in name_lower:
                return True
        
        return False
    
    def _deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Remove duplicate relationships"""
        seen = set()
        deduplicated = []
        
        for rel in relationships:
            # Create unique key
            key = (
                rel['type'],
                rel['from_entity'].lower(),
                rel['to_entity'].lower()
            )
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(rel)
        
        return deduplicated


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def map_relationships_from_entities(
    entities: Dict[str, List[Dict]],
    text: str
) -> List[Dict]:
    """
    Convenience function for quick relationship mapping
    
    Args:
        entities: Extracted entities
        text: Document text
        
    Returns:
        List of relationships
    """
    mapper = RelationshipMapper()
    return mapper.map_relationships(entities, text)


if __name__ == "__main__":
    # Test the relationship mapper
    from core.entity_extractor import EntityExtractor
    
    test_text = """
    CFC Corporation sold $4.26 million of holographic stripe products to Visa Inc. 
    in March 2006. However, a directive from Visa on March 15, 2006 required CFC 
    to discontinue production. This represents 100% of CFC's revenue from this 
    product line. The company had $346,000 in inventory at the time.
    
    The 10-K filing described the product as "growing" while the Visa email 
    indicated production must stop immediately.
    """
    
    print("="*80)
    print("RELATIONSHIP MAPPER TEST")
    print("="*80)
    
    # First extract entities
    extractor = EntityExtractor()
    entities = extractor.extract_entities(test_text)
    
    print("\n📊 Extracted Entities:")
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"  {entity_type}: {len(entity_list)}")
    
    # Then map relationships
    mapper = RelationshipMapper()
    relationships = mapper.map_relationships(entities, test_text)
    
    print(f"\n🔗 Mapped Relationships: {len(relationships)}")
    print("-" * 80)
    
    for rel in relationships:
        print(f"\n{rel['type']}:")
        print(f"  From: {rel['from_entity']} ({rel['from_type']})")
        print(f"  To: {rel['to_entity']} ({rel['to_type']})")
        if rel['properties']:
            print(f"  Properties: {rel['properties']}")