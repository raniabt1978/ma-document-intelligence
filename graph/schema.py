# graph/schema.py
"""
Knowledge Graph Schema Definition for M&A Document Intelligence
Defines all node types, relationship types, constraints, and indexes
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS FOR CONTROLLED VOCABULARIES
# ============================================================================

class NodeLabel(str, Enum):
    """All possible node types in the knowledge graph"""
    ORGANIZATION = "Organization"
    PRODUCT = "Product"
    FINANCIAL_METRIC = "FinancialMetric"
    PERSON = "Person"
    EVENT = "Event"
    DOCUMENT = "Document"
    RISK = "Risk"
    DATE = "Date"
    LOCATION = "Location"


class RelationshipType(str, Enum):
    """All possible relationship types in the knowledge graph"""
    # Business relationships
    SELLS_TO = "SELLS_TO"
    BUYS_FROM = "BUYS_FROM"
    PRODUCES = "PRODUCES"
    ACQUIRING = "ACQUIRING"
    ACQUIRED_BY = "ACQUIRED_BY"
    COMPETES_WITH = "COMPETES_WITH"
    SUPPLIES_TO = "SUPPLIES_TO"
    PARTNERS_WITH = "PARTNERS_WITH"
    
    # Dependencies
    DEPENDS_ON = "DEPENDS_ON"
    RELIES_ON = "RELIES_ON"
    REQUIRES = "REQUIRES"
    
    # Financial relationships
    HAS_METRIC = "HAS_METRIC"
    GENERATES_REVENUE = "GENERATES_REVENUE"
    INCURS_COST = "INCURS_COST"
    
    # Events and impacts
    AFFECTS = "AFFECTS"
    IMPACTS = "IMPACTS"
    CREATES = "CREATES"
    TRIGGERS = "TRIGGERS"
    ISSUES = "ISSUES"
    RECEIVES = "RECEIVES"
    
    # Contradictions (CRITICAL for M&A)
    CONTRADICTS = "CONTRADICTS"
    CONFLICTS_WITH = "CONFLICTS_WITH"
    INVALIDATES = "INVALIDATES"
    
    # Document relationships
    MENTIONS = "MENTIONS"
    CONTAINS = "CONTAINS"
    DESCRIBES = "DESCRIBES"
    REFERENCES = "REFERENCES"
    AUTHORED_BY = "AUTHORED_BY"
    
    # Temporal relationships
    OCCURS_BEFORE = "OCCURS_BEFORE"
    OCCURS_AFTER = "OCCURS_AFTER"
    HAPPENS_ON = "HAPPENS_ON"
    
    # Organizational relationships
    EMPLOYED_BY = "EMPLOYED_BY"
    MANAGES = "MANAGES"
    LOCATED_IN = "LOCATED_IN"


class Severity(str, Enum):
    """Risk severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class OrganizationType(str, Enum):
    """Types of organizations"""
    TARGET = "target"
    ACQUIRER = "acquirer"
    CUSTOMER = "customer"
    SUPPLIER = "supplier"
    COMPETITOR = "competitor"
    PARTNER = "partner"


class ProductStatus(str, Enum):
    """Product lifecycle status"""
    ACTIVE = "active"
    DISCONTINUED = "discontinued"
    GROWING = "growing"
    DECLINING = "declining"
    PLANNED = "planned"


class EventType(str, Enum):
    """Types of events"""
    DIRECTIVE = "directive"
    FILING = "filing"
    COMMUNICATION = "communication"
    TRANSACTION = "transaction"
    MEETING = "meeting"
    ANNOUNCEMENT = "announcement"


class RiskType(str, Enum):
    """Types of risks"""
    CONCENTRATION = "concentration"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    LEGAL = "legal"
    STRATEGIC = "strategic"
    COMPLIANCE = "compliance"
    MARKET = "market"


# ============================================================================
# SCHEMA DEFINITIONS
# ============================================================================

@dataclass
class NodeSchema:
    """Schema definition for a node type"""
    label: str
    properties: Dict[str, str]  # property_name: property_type
    required_properties: List[str]
    indexed_properties: List[str]
    unique_properties: List[str]
    description: str


@dataclass
class RelationshipSchema:
    """Schema definition for a relationship type"""
    type: str
    from_labels: List[str]  # Possible source node labels
    to_labels: List[str]    # Possible target node labels
    properties: Dict[str, str]
    description: str


# ============================================================================
# NODE SCHEMAS
# ============================================================================

NODE_SCHEMAS = {
    NodeLabel.ORGANIZATION: NodeSchema(
        label="Organization",
        properties={
            "id": "string",
            "name": "string",
            "type": "string",  # target, acquirer, customer, supplier
            "industry": "string",
            "revenue": "float",
            "employees": "integer",
            "description": "string",
            "confidence": "float",
            "created_at": "datetime",
            "updated_at": "datetime"
        },
        required_properties=["id", "name"],
        indexed_properties=["name", "type", "industry"],
        unique_properties=["id"],
        description="Companies, customers, suppliers, competitors"
    ),
    
    NodeLabel.PRODUCT: NodeSchema(
        label="Product",
        properties={
            "id": "string",
            "name": "string",
            "category": "string",
            "status": "string",  # active, discontinued, growing, declining
            "revenue_contribution": "float",
            "description": "string",
            "confidence": "float",
            "created_at": "datetime",
            "updated_at": "datetime"
        },
        required_properties=["id", "name"],
        indexed_properties=["name", "status", "category"],
        unique_properties=["id"],
        description="Products and services offered by companies"
    ),
    
    NodeLabel.FINANCIAL_METRIC: NodeSchema(
        label="FinancialMetric",
        properties={
            "id": "string",
            "name": "string",
            "value": "float",
            "unit": "string",  # USD, percentage, ratio
            "year": "integer",
            "quarter": "string",
            "period": "string",
            "source_doc": "string",
            "confidence": "float",
            "created_at": "datetime"
        },
        required_properties=["id", "name", "value"],
        indexed_properties=["name", "year", "quarter"],
        unique_properties=["id"],
        description="Financial metrics, ratios, and KPIs"
    ),
    
    NodeLabel.PERSON: NodeSchema(
        label="Person",
        properties={
            "id": "string",
            "name": "string",
            "role": "string",
            "title": "string",
            "email": "string",
            "organization": "string",
            "confidence": "float",
            "created_at": "datetime"
        },
        required_properties=["id", "name"],
        indexed_properties=["name", "role", "email"],
        unique_properties=["id"],
        description="People mentioned in documents (executives, analysts, etc.)"
    ),
    
    NodeLabel.EVENT: NodeSchema(
        label="Event",
        properties={
            "id": "string",
            "type": "string",  # directive, filing, communication, transaction
            "date": "datetime",
            "description": "string",
            "impact": "string",  # critical, high, medium, low
            "source_doc": "string",
            "confidence": "float",
            "created_at": "datetime"
        },
        required_properties=["id", "type", "description"],
        indexed_properties=["type", "date", "impact"],
        unique_properties=["id"],
        description="Events like directives, filings, communications"
    ),
    
    NodeLabel.DOCUMENT: NodeSchema(
        label="Document",
        properties={
            "id": "string",
            "filename": "string",
            "doc_type": "string",  # pdf, excel, email, url
            "upload_date": "datetime",
            "word_count": "integer",
            "chunks": "integer",
            "vector_id": "string",  # Link to vector store
            "url": "string",
            "created_at": "datetime"
        },
        required_properties=["id", "filename", "doc_type"],
        indexed_properties=["filename", "doc_type", "upload_date"],
        unique_properties=["id"],
        description="Source documents in the system"
    ),
    
    NodeLabel.RISK: NodeSchema(
        label="Risk",
        properties={
            "id": "string",
            "type": "string",  # concentration, operational, financial, legal
            "severity": "string",  # critical, high, medium, low
            "description": "string",
            "confidence": "float",
            "detected_date": "datetime",
            "mitigation": "string",
            "created_at": "datetime"
        },
        required_properties=["id", "type", "severity", "description"],
        indexed_properties=["type", "severity", "detected_date"],
        unique_properties=["id"],
        description="Identified risks in M&A analysis"
    ),
    
    NodeLabel.DATE: NodeSchema(
        label="Date",
        properties={
            "id": "string",
            "date": "datetime",
            "year": "integer",
            "quarter": "string",
            "month": "string",
            "display": "string"
        },
        required_properties=["id", "date"],
        indexed_properties=["date", "year", "quarter"],
        unique_properties=["id"],
        description="Temporal nodes for timeline analysis"
    ),
    
    NodeLabel.LOCATION: NodeSchema(
        label="Location",
        properties={
            "id": "string",
            "name": "string",
            "type": "string",  # city, state, country, region
            "country": "string",
            "confidence": "float"
        },
        required_properties=["id", "name"],
        indexed_properties=["name", "type", "country"],
        unique_properties=["id"],
        description="Geographic locations"
    )
}


# ============================================================================
# RELATIONSHIP SCHEMAS
# ============================================================================

RELATIONSHIP_SCHEMAS = {
    RelationshipType.SELLS_TO: RelationshipSchema(
        type="SELLS_TO",
        from_labels=["Organization"],
        to_labels=["Organization"],
        properties={
            "volume": "float",
            "percentage": "float",  # % of seller's revenue
            "value": "float",       # Dollar amount
            "start_date": "datetime",
            "end_date": "datetime",
            "confidence": "float"
        },
        description="Company sells products/services to another company"
    ),
    
    RelationshipType.PRODUCES: RelationshipSchema(
        type="PRODUCES",
        from_labels=["Organization"],
        to_labels=["Product"],
        properties={
            "volume": "integer",
            "revenue": "float",
            "start_date": "datetime",
            "confidence": "float"
        },
        description="Company produces a product"
    ),
    
    RelationshipType.ACQUIRING: RelationshipSchema(
        type="ACQUIRING",
        from_labels=["Organization"],
        to_labels=["Organization"],
        properties={
            "amount": "float",
            "date": "datetime",
            "status": "string",  # pending, completed, failed
            "deal_type": "string",
            "confidence": "float"
        },
        description="Company acquiring another company"
    ),
    
    RelationshipType.DEPENDS_ON: RelationshipSchema(
        type="DEPENDS_ON",
        from_labels=["Product", "Organization"],
        to_labels=["Organization", "Product"],
        properties={
            "criticality": "string",  # critical, high, medium, low
            "percentage": "float",
            "description": "string",
            "confidence": "float"
        },
        description="Dependency relationship indicating reliance"
    ),
    
    RelationshipType.HAS_METRIC: RelationshipSchema(
        type="HAS_METRIC",
        from_labels=["Organization", "Product"],
        to_labels=["FinancialMetric"],
        properties={
            "source_doc": "string",
            "confidence": "float"
        },
        description="Entity has a financial metric"
    ),
    
    RelationshipType.GENERATES_REVENUE: RelationshipSchema(
        type="GENERATES_REVENUE",
        from_labels=["Product", "Organization"],
        to_labels=["FinancialMetric"],
        properties={
            "percentage": "float",
            "confidence": "float"
        },
        description="Product/organization generates revenue"
    ),
    
    RelationshipType.AFFECTS: RelationshipSchema(
        type="AFFECTS",
        from_labels=["Event"],
        to_labels=["Product", "Organization", "FinancialMetric"],
        properties={
            "impact_type": "string",
            "severity": "string",
            "description": "string",
            "confidence": "float"
        },
        description="Event affects an entity"
    ),
    
    RelationshipType.CREATES: RelationshipSchema(
        type="CREATES",
        from_labels=["Event", "Document"],
        to_labels=["Risk"],
        properties={
            "confidence": "float"
        },
        description="Event or finding creates a risk"
    ),
    
    RelationshipType.CONTRADICTS: RelationshipSchema(
        type="CONTRADICTS",
        from_labels=["Document", "FinancialMetric", "Event"],
        to_labels=["Document", "FinancialMetric", "Event"],
        properties={
            "aspect": "string",
            "severity": "string",
            "explanation": "string",
            "confidence": "float"
        },
        description="CRITICAL: Contradictory information between entities"
    ),
    
    RelationshipType.MENTIONS: RelationshipSchema(
        type="MENTIONS",
        from_labels=["Document"],
        to_labels=["Organization", "Product", "Person", "Event"],
        properties={
            "count": "integer",
            "context": "string",
            "confidence": "float"
        },
        description="Document mentions an entity"
    ),
    
    RelationshipType.CONTAINS: RelationshipSchema(
        type="CONTAINS",
        from_labels=["Document"],
        to_labels=["FinancialMetric", "Event", "Risk"],
        properties={
            "page": "integer",
            "section": "string",
            "confidence": "float"
        },
        description="Document contains specific information"
    ),
    
    RelationshipType.AUTHORED_BY: RelationshipSchema(
        type="AUTHORED_BY",
        from_labels=["Document", "Event"],
        to_labels=["Person", "Organization"],
        properties={
            "date": "datetime",
            "confidence": "float"
        },
        description="Document or event authored by person/organization"
    ),
    
    RelationshipType.OCCURS_BEFORE: RelationshipSchema(
        type="OCCURS_BEFORE",
        from_labels=["Event"],
        to_labels=["Event"],
        properties={
            "days_difference": "integer"
        },
        description="Temporal ordering of events"
    ),
    
    RelationshipType.INVALIDATES: RelationshipSchema(
        type="INVALIDATES",
        from_labels=["Event"],
        to_labels=["Document", "FinancialMetric"],
        properties={
            "reason": "string",
            "confidence": "float"
        },
        description="Event makes previous information invalid"
    )
}


# ============================================================================
# CONSTRAINTS AND INDEXES
# ============================================================================

def get_constraints() -> List[str]:
    """
    Get list of constraint statements to create
    Ensures data integrity in the graph
    """
    constraints = []
    
    for label, schema in NODE_SCHEMAS.items():
        for prop in schema.unique_properties:
            # Unique constraint
            constraint = f"""
            CREATE CONSTRAINT {label.value.lower()}_{prop}_unique IF NOT EXISTS
            FOR (n:{label.value})
            REQUIRE n.{prop} IS UNIQUE
            """
            constraints.append(constraint.strip())
        
        # Ensure required properties exist
        for prop in schema.required_properties:
            constraint = f"""
            CREATE CONSTRAINT {label.value.lower()}_{prop}_exists IF NOT EXISTS
            FOR (n:{label.value})
            REQUIRE n.{prop} IS NOT NULL
            """
            constraints.append(constraint.strip())
    
    return constraints


def get_indexes() -> List[str]:
    """
    Get list of index statements to create
    Improves query performance
    """
    indexes = []
    
    for label, schema in NODE_SCHEMAS.items():
        for prop in schema.indexed_properties:
            index = f"""
            CREATE INDEX {label.value.lower()}_{prop}_index IF NOT EXISTS
            FOR (n:{label.value})
            ON (n.{prop})
            """
            indexes.append(index.strip())
    
    return indexes


def get_full_text_indexes() -> List[str]:
    """
    Get full-text search indexes for text properties
    Enables advanced text search
    """
    indexes = [
        """
        CREATE FULLTEXT INDEX organization_search IF NOT EXISTS
        FOR (n:Organization)
        ON EACH [n.name, n.description, n.industry]
        """,
        """
        CREATE FULLTEXT INDEX product_search IF NOT EXISTS
        FOR (n:Product)
        ON EACH [n.name, n.description, n.category]
        """,
        """
        CREATE FULLTEXT INDEX person_search IF NOT EXISTS
        FOR (n:Person)
        ON EACH [n.name, n.role, n.title]
        """,
        """
        CREATE FULLTEXT INDEX event_search IF NOT EXISTS
        FOR (n:Event)
        ON EACH [n.description, n.type]
        """,
        """
        CREATE FULLTEXT INDEX risk_search IF NOT EXISTS
        FOR (n:Risk)
        ON EACH [n.description, n.type, n.mitigation]
        """
    ]
    
    return [idx.strip() for idx in indexes]


# ============================================================================
# SCHEMA VALIDATION
# ============================================================================

def validate_node_data(label: str, data: Dict[str, Any]) -> bool:
    """
    Validate node data against schema
    
    Args:
        label: Node label (e.g., "Organization")
        data: Node properties
        
    Returns:
        bool: True if valid
    """
    if label not in NODE_SCHEMAS:
        logger.error(f"Unknown node label: {label}")
        return False
    
    schema = NODE_SCHEMAS[label]
    
    # Check required properties
    for prop in schema.required_properties:
        if prop not in data:
            logger.error(f"Missing required property '{prop}' for {label}")
            return False
    
    return True


def validate_relationship_data(rel_type: str, from_label: str, to_label: str) -> bool:
    """
    Validate relationship between node types
    
    Args:
        rel_type: Relationship type
        from_label: Source node label
        to_label: Target node label
        
    Returns:
        bool: True if valid
    """
    if rel_type not in RELATIONSHIP_SCHEMAS:
        logger.error(f"Unknown relationship type: {rel_type}")
        return False
    
    schema = RELATIONSHIP_SCHEMAS[rel_type]
    
    if from_label not in schema.from_labels:
        logger.error(f"Invalid source label '{from_label}' for {rel_type}")
        return False
    
    if to_label not in schema.to_labels:
        logger.error(f"Invalid target label '{to_label}' for {rel_type}")
        return False
    
    return True


# ============================================================================
# SCHEMA DOCUMENTATION
# ============================================================================

def print_schema_documentation():
    """Print comprehensive schema documentation"""
    print("\n" + "="*80)
    print("M&A KNOWLEDGE GRAPH SCHEMA DOCUMENTATION")
    print("="*80)
    
    print("\n📊 NODE TYPES:")
    print("-" * 80)
    for label, schema in NODE_SCHEMAS.items():
        print(f"\n{label.value}:")
        print(f"  Description: {schema.description}")
        print(f"  Required: {', '.join(schema.required_properties)}")
        print(f"  Indexed: {', '.join(schema.indexed_properties)}")
        print(f"  Unique: {', '.join(schema.unique_properties)}")
    
    print("\n\n🔗 RELATIONSHIP TYPES:")
    print("-" * 80)
    for rel_type, schema in RELATIONSHIP_SCHEMAS.items():
        print(f"\n{rel_type.value}:")
        print(f"  Description: {schema.description}")
        print(f"  From: {', '.join(schema.from_labels)}")
        print(f"  To: {', '.join(schema.to_labels)}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Print schema documentation
    print_schema_documentation()
    
    # Show constraint and index creation statements
    print("\n\n🔒 CONSTRAINTS TO CREATE:")
    for constraint in get_constraints()[:3]:  # Show first 3
        print(f"  {constraint[:100]}...")
    print(f"  ... and {len(get_constraints())-3} more")
    
    print("\n\n🔍 INDEXES TO CREATE:")
    for index in get_indexes()[:3]:  # Show first 3
        print(f"  {index[:100]}...")
    print(f"  ... and {len(get_indexes())-3} more")