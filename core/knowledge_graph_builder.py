# core/knowledge_graph_builder.py
"""
Knowledge Graph Builder
Builds Neo4j knowledge graph from extracted entities and relationships
"""

import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from core.neo4j_client import Neo4jClient, get_neo4j_client
from core.entity_extractor import EntityExtractor
from core.relationship_mapper import RelationshipMapper
from graph.schema import (
    NODE_SCHEMAS,
    NodeLabel,
    get_constraints,
    get_indexes,
    validate_node_data
)

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """
    Builds and maintains the knowledge graph in Neo4j
    Converts extracted entities and relationships into graph nodes and edges
    """
    
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        """
        Initialize knowledge graph builder
        
        Args:
            neo4j_client: Neo4j client instance (creates new if None)
        """
        self.neo4j = neo4j_client or get_neo4j_client()
        
        # Initialize extractors
        self.entity_extractor = EntityExtractor()
        self.relationship_mapper = RelationshipMapper()
        
        # Ensure schema is set up
        self._setup_schema()
    
    def _setup_schema(self):
        """Set up Neo4j schema (constraints and indexes)"""
        try:
            logger.info("Setting up Neo4j schema...")
            
            # Create constraints
            constraints = get_constraints()
            self.neo4j.create_constraints(constraints)
            
            # Create indexes
            indexes = get_indexes()
            self.neo4j.create_indexes(indexes)
            
            logger.info("Schema setup complete")
            
        except Exception as e:
            logger.warning(f"Schema setup warning: {e}")
            # Continue even if some constraints/indexes already exist
    
    def build_graph_from_document(
        self,
        text: str,
        document_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build knowledge graph from a single document
        
        Args:
            text: Document text
            document_metadata: Document metadata (filename, doc_type, etc.)
            
        Returns:
            Dictionary with build statistics
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Building graph from document: {document_metadata.get('filename')}")
            
            # Step 1: Extract entities
            entities = self.entity_extractor.extract_entities(
                text,
                source_doc=document_metadata.get('filename')
            )
            
            # Step 2: Map relationships
            relationships = self.relationship_mapper.map_relationships(entities, text)
            
            # Step 3: Create document node
            doc_node = self._create_document_node(document_metadata)
            
            # Step 4: Create entity nodes
            entity_stats = self._create_entity_nodes(entities, doc_node['id'])
            
            # Step 5: Create relationships
            rel_stats = self._create_relationship_edges(relationships)
            
            # Step 6: Link entities to document
            self._link_entities_to_document(entities, doc_node['id'])
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            stats = {
                'status': 'success',
                'document_id': doc_node['id'],
                'entities_created': entity_stats,
                'relationships_created': rel_stats,
                'processing_time_seconds': processing_time
            }
            
            logger.info(f"Graph built successfully: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _create_document_node(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Create Document node in graph"""
        doc_id = self._generate_id(metadata.get('filename', 'unknown'))
        
        query = """
        MERGE (d:Document {id: $id})
        SET d.filename = $filename,
            d.doc_type = $doc_type,
            d.upload_date = $upload_date,
            d.word_count = $word_count,
            d.chunks = $chunks,
            d.created_at = datetime()
        RETURN d.id as id
        """
        
        params = {
            'id': doc_id,
            'filename': metadata.get('filename', 'Unknown'),
            'doc_type': metadata.get('doc_type', 'unknown'),
            'upload_date': metadata.get('upload_date', datetime.utcnow().isoformat()),
            'word_count': metadata.get('word_count', 0),
            'chunks': metadata.get('chunks', 0)
        }
        
        result = self.neo4j.execute_write(query, params)
        
        return {'id': doc_id}
    
    def _create_entity_nodes(
        self,
        entities: Dict[str, List[Dict]],
        doc_id: str
    ) -> Dict[str, int]:
        """Create nodes for all extracted entities"""
        stats = {}
        
        # Create organization nodes
        stats['organizations'] = self._create_organization_nodes(
            entities.get('organizations', [])
        )
        
        # Create product nodes
        stats['products'] = self._create_product_nodes(
            entities.get('products', [])
        )
        
        # Create financial metric nodes
        stats['financial_metrics'] = self._create_financial_metric_nodes(
            entities.get('financial_metrics', [])
        )
        
        # Create person nodes
        stats['people'] = self._create_person_nodes(
            entities.get('people', [])
        )
        
        # Create event nodes
        stats['events'] = self._create_event_nodes(
            entities.get('events', [])
        )
        
        # Create date nodes
        stats['dates'] = self._create_date_nodes(
            entities.get('dates', [])
        )
        
        # Create location nodes
        stats['locations'] = self._create_location_nodes(
            entities.get('locations', [])
        )
        
        return stats
    
    def _create_organization_nodes(self, organizations: List[Dict]) -> int:
        """Create Organization nodes"""
        count = 0
        
        for org in organizations:
            org_id = self._generate_id(org['name'])
            
            query = """
            MERGE (o:Organization {id: $id})
            SET o.name = $name,
                o.type = $type,
                o.confidence = $confidence,
                o.created_at = datetime(),
                o.updated_at = datetime()
            RETURN o.id as id
            """
            
            params = {
                'id': org_id,
                'name': org['name'],
                'type': org.get('type', 'target'),
                'confidence': org.get('confidence', 0.85)
            }
            
            try:
                self.neo4j.execute_write(query, params)
                count += 1
            except Exception as e:
                logger.error(f"Error creating organization node: {e}")
        
        return count
    
    def _create_product_nodes(self, products: List[Dict]) -> int:
        """Create Product nodes"""
        count = 0
        
        for product in products:
            product_id = self._generate_id(product['name'])
            
            query = """
            MERGE (p:Product {id: $id})
            SET p.name = $name,
                p.status = $status,
                p.confidence = $confidence,
                p.created_at = datetime(),
                p.updated_at = datetime()
            RETURN p.id as id
            """
            
            params = {
                'id': product_id,
                'name': product['name'],
                'status': product.get('status', 'active'),
                'confidence': product.get('confidence', 0.70)
            }
            
            try:
                self.neo4j.execute_write(query, params)
                count += 1
            except Exception as e:
                logger.error(f"Error creating product node: {e}")
        
        return count
    
    def _create_financial_metric_nodes(self, metrics: List[Dict]) -> int:
        """Create FinancialMetric nodes"""
        count = 0
        
        for metric in metrics:
            metric_id = self._generate_id(f"{metric['name']}_{metric['value']}")
            
            query = """
            MERGE (f:FinancialMetric {id: $id})
            SET f.name = $name,
                f.value = $value,
                f.unit = $unit,
                f.confidence = $confidence,
                f.created_at = datetime()
            RETURN f.id as id
            """
            
            params = {
                'id': metric_id,
                'name': metric['name'],
                'value': float(metric['value']),
                'unit': metric.get('unit', 'USD'),
                'confidence': metric.get('confidence', 0.90)
            }
            
            try:
                self.neo4j.execute_write(query, params)
                count += 1
            except Exception as e:
                logger.error(f"Error creating financial metric node: {e}")
        
        return count
    
    def _create_person_nodes(self, people: List[Dict]) -> int:
        """Create Person nodes"""
        count = 0
        
        for person in people:
            person_id = self._generate_id(person['name'])
            
            query = """
            MERGE (p:Person {id: $id})
            SET p.name = $name,
                p.confidence = $confidence,
                p.created_at = datetime()
            RETURN p.id as id
            """
            
            params = {
                'id': person_id,
                'name': person['name'],
                'confidence': person.get('confidence', 0.85)
            }
            
            try:
                self.neo4j.execute_write(query, params)
                count += 1
            except Exception as e:
                logger.error(f"Error creating person node: {e}")
        
        return count
    
    def _create_event_nodes(self, events: List[Dict]) -> int:
        """Create Event nodes"""
        count = 0
        
        for event in events:
            event_id = self._generate_id(event['description'][:50])
            
            query = """
            MERGE (e:Event {id: $id})
            SET e.type = $type,
                e.description = $description,
                e.confidence = $confidence,
                e.created_at = datetime()
            RETURN e.id as id
            """
            
            params = {
                'id': event_id,
                'type': event.get('type', 'unknown'),
                'description': event['description'][:200],  # Limit length
                'confidence': event.get('confidence', 0.75)
            }
            
            try:
                self.neo4j.execute_write(query, params)
                count += 1
            except Exception as e:
                logger.error(f"Error creating event node: {e}")
        
        return count
    
    def _create_date_nodes(self, dates: List[Dict]) -> int:
        """Create Date nodes"""
        count = 0
        
        for date in dates:
            if not date.get('parsed'):
                continue
            
            date_id = self._generate_id(date['parsed'])
            
            query = """
            MERGE (d:Date {id: $id})
            SET d.date = datetime($date),
                d.display = $display,
                d.confidence = $confidence
            RETURN d.id as id
            """
            
            params = {
                'id': date_id,
                'date': date['parsed'],
                'display': date['original'],
                'confidence': date.get('confidence', 0.90)
            }
            
            try:
                self.neo4j.execute_write(query, params)
                count += 1
            except Exception as e:
                logger.error(f"Error creating date node: {e}")
        
        return count
    
    def _create_location_nodes(self, locations: List[Dict]) -> int:
        """Create Location nodes"""
        count = 0
        
        for location in locations:
            location_id = self._generate_id(location['name'])
            
            query = """
            MERGE (l:Location {id: $id})
            SET l.name = $name,
                l.type = $type,
                l.confidence = $confidence
            RETURN l.id as id
            """
            
            params = {
                'id': location_id,
                'name': location['name'],
                'type': location.get('type', 'location'),
                'confidence': location.get('confidence', 0.80)
            }
            
            try:
                self.neo4j.execute_write(query, params)
                count += 1
            except Exception as e:
                logger.error(f"Error creating location node: {e}")
        
        return count
    
    def _create_relationship_edges(self, relationships: List[Dict]) -> int:
        """Create relationship edges between nodes"""
        count = 0
        
        for rel in relationships:
            try:
                # Generate node IDs
                from_id = self._generate_id(rel['from_entity'])
                to_id = self._generate_id(rel['to_entity'])
                
                # Build dynamic Cypher query
                rel_type = rel['type'].upper().replace(' ', '_')
                
                query = f"""
                MATCH (from {{id: $from_id}})
                MATCH (to {{id: $to_id}})
                MERGE (from)-[r:{rel_type}]->(to)
                SET r.confidence = $confidence,
                    r.created_at = datetime()
                """
                
                # Add additional properties
                if rel.get('properties'):
                    for key, value in rel['properties'].items():
                        if key not in ['confidence']:  # Skip already set
                            query += f", r.{key} = ${key}"
                
                query += " RETURN r"
                
                # Prepare parameters
                params = {
                    'from_id': from_id,
                    'to_id': to_id,
                    'confidence': rel['properties'].get('confidence', 0.80)
                }
                
                # Add property parameters
                if rel.get('properties'):
                    for key, value in rel['properties'].items():
                        if key not in ['confidence']:
                            params[key] = value
                
                self.neo4j.execute_write(query, params)
                count += 1
                
            except Exception as e:
                logger.error(f"Error creating relationship: {e}")
                logger.debug(f"Failed relationship: {rel}")
        
        return count
    
    def _link_entities_to_document(
        self,
        entities: Dict[str, List[Dict]],
        doc_id: str
    ):
        """Create MENTIONS relationships from document to entities"""
        
        # Link organizations
        for org in entities.get('organizations', []):
            org_id = self._generate_id(org['name'])
            self._create_mentions_link(doc_id, org_id)
        
        # Link products
        for product in entities.get('products', []):
            product_id = self._generate_id(product['name'])
            self._create_mentions_link(doc_id, product_id)
        
        # Link people
        for person in entities.get('people', []):
            person_id = self._generate_id(person['name'])
            self._create_mentions_link(doc_id, person_id)
    
    def _create_mentions_link(self, doc_id: str, entity_id: str):
        """Create MENTIONS relationship"""
        query = """
        MATCH (d:Document {id: $doc_id})
        MATCH (e {id: $entity_id})
        MERGE (d)-[r:MENTIONS]->(e)
        SET r.created_at = datetime()
        """
        
        try:
            self.neo4j.execute_write(query, {
                'doc_id': doc_id,
                'entity_id': entity_id
            })
        except Exception as e:
            logger.debug(f"Error creating MENTIONS link: {e}")
    
    def _generate_id(self, text: str) -> str:
        """Generate consistent ID from text"""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get current knowledge graph statistics"""
        return self.neo4j.get_stats()
    
    def clear_graph(self, confirm: bool = False):
        """Clear entire knowledge graph - USE WITH CAUTION!"""
        if not confirm:
            raise ValueError("Must set confirm=True to clear graph")
        
        logger.warning("⚠️  CLEARING ENTIRE KNOWLEDGE GRAPH")
        self.neo4j.clear_database(confirm=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_graph_from_text(
    text: str,
    filename: str,
    doc_type: str = "unknown"
) -> Dict[str, Any]:
    """
    Convenience function to build graph from text
    
    Args:
        text: Document text
        filename: Document filename
        doc_type: Document type
        
    Returns:
        Build statistics
    """
    builder = KnowledgeGraphBuilder()
    
    metadata = {
        'filename': filename,
        'doc_type': doc_type,
        'word_count': len(text.split()),
        'upload_date': datetime.utcnow().isoformat()
    }
    
    return builder.build_graph_from_document(text, metadata)


if __name__ == "__main__":
    # Test the knowledge graph builder
    test_text = """
    CFC Corporation sold $4.26 million of holographic stripe products to Visa Inc. 
    in March 2006. However, a directive from Visa on March 15, 2006 required CFC 
    to discontinue production. This represents 100% of CFC's revenue from this 
    product line. The company had $346,000 in inventory at the time.
    
    The 10-K filing described the product as "growing" while the Visa email 
    indicated production must stop immediately.
    
    Contact: john.doe@cfc.com for more information.
    """
    
    print("="*80)
    print("KNOWLEDGE GRAPH BUILDER TEST")
    print("="*80)
    
    try:
        # Build graph from test text
        builder = KnowledgeGraphBuilder()
        
        metadata = {
            'filename': 'test_cfc_document.pdf',
            'doc_type': 'pdf',
            'word_count': len(test_text.split()),
            'upload_date': datetime.utcnow().isoformat()
        }
        
        print("\n🏗️  Building knowledge graph...")
        stats = builder.build_graph_from_document(test_text, metadata)
        
        print("\n✅ Graph Build Complete!")
        print("-" * 80)
        print(f"Status: {stats['status']}")
        print(f"Document ID: {stats.get('document_id')}")
        print(f"\nEntities Created:")
        for entity_type, count in stats.get('entities_created', {}).items():
            print(f"  {entity_type}: {count}")
        print(f"\nRelationships Created: {stats.get('relationships_created', 0)}")
        print(f"Processing Time: {stats.get('processing_time_seconds', 0):.2f}s")
        
        # Show graph stats
        print("\n📊 Current Graph Statistics:")
        graph_stats = builder.get_graph_stats()
        print(f"  Total Nodes: {graph_stats['total_nodes']}")
        print(f"  Total Relationships: {graph_stats['total_relationships']}")
        
        if graph_stats.get('node_labels'):
            print("\n  Node Types:")
            for label, count in graph_stats['node_labels'].items():
                print(f"    {label}: {count}")
        
        if graph_stats.get('relationship_types'):
            print("\n  Relationship Types:")
            for rel_type, count in graph_stats['relationship_types'].items():
                print(f"    {rel_type}: {count}")
        
        print("\n🌐 View in Neo4j Browser: http://localhost:7474")
        print("   Run query: MATCH (n) RETURN n LIMIT 25")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()