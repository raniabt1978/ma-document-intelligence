# core/neo4j_client.py
"""
Production Neo4j Client for Knowledge Graph
Handles connections, transactions, health checks, and error handling
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
from datetime import datetime
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from neo4j import GraphDatabase, Driver, Session, Transaction
from neo4j.exceptions import (
    ServiceUnavailable, 
    AuthError, 
    ConfigurationError,
    TransientError
)

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Production-grade Neo4j client with connection pooling and error handling"""
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = "neo4j",
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50,
        connection_timeout: int = 30
    ):
        """
        Initialize Neo4j client
        
        Args:
            uri: Neo4j connection URI (bolt://localhost:7687)
            username: Database username
            password: Database password
            database: Database name (default: neo4j)
            max_connection_lifetime: Max seconds for connection (default: 3600)
            max_connection_pool_size: Max connections in pool (default: 50)
            connection_timeout: Connection timeout in seconds (default: 30)
        """
        # Get credentials from environment or parameters
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        
        # Connection settings
        self.max_connection_lifetime = max_connection_lifetime
        self.max_connection_pool_size = max_connection_pool_size
        self.connection_timeout = connection_timeout
        
        # Initialize driver
        self.driver: Optional[Driver] = None
        self._is_connected = False
        
        # Retry settings
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # Initialize connection
        self.connect()
    
    def connect(self) -> bool:
        """
        Establish connection to Neo4j with retry logic
        
        Returns:
            bool: True if connected successfully
        """
        if self._is_connected and self.driver:
            logger.info("Neo4j client already connected")
            return True
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Connecting to Neo4j at {self.uri} (attempt {attempt + 1}/{self.max_retries})")
                
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password),
                    max_connection_lifetime=self.max_connection_lifetime,
                    max_connection_pool_size=self.max_connection_pool_size,
                    connection_timeout=self.connection_timeout
                )
                
                # Verify connectivity
                self.driver.verify_connectivity()
                
                self._is_connected = True
                logger.info(f"✅ Successfully connected to Neo4j at {self.uri}")
                
                # Log connection info
                server_info = self.get_server_info()
                logger.info(f"Neo4j version: {server_info.get('version', 'unknown')}")
                
                return True
                
            except AuthError as e:
                logger.error(f"❌ Authentication failed: {e}")
                raise  # Don't retry auth errors
                
            except (ServiceUnavailable, ConfigurationError) as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (attempt + 1)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Failed to connect to Neo4j after {self.max_retries} attempts")
                    raise
                    
            except Exception as e:
                logger.error(f"❌ Unexpected error connecting to Neo4j: {e}")
                raise
        
        return False
    
    def disconnect(self):
        """Close Neo4j connection"""
        if self.driver:
            try:
                self.driver.close()
                self._is_connected = False
                logger.info("Neo4j connection closed")
            except Exception as e:
                logger.error(f"Error closing Neo4j connection: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to Neo4j"""
        if not self._is_connected or not self.driver:
            return False
        
        try:
            self.driver.verify_connectivity()
            return True
        except:
            self._is_connected = False
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Neo4j connection
        
        Returns:
            Dict with health status
        """
        health = {
            "status": "unhealthy",
            "connected": False,
            "timestamp": datetime.utcnow().isoformat(),
            "uri": self.uri,
            "database": self.database
        }
        
        try:
            if not self.is_connected():
                self.connect()
            
            # Test query
            result = self.execute_read("RETURN 1 as test")
            
            if result and result[0].get("test") == 1:
                health["status"] = "healthy"
                health["connected"] = True
                
                # Get database stats
                stats = self.get_stats()
                health["stats"] = stats
                
        except Exception as e:
            health["error"] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health
    
    @contextmanager
    def session(self, database: Optional[str] = None):
        """
        Context manager for Neo4j session
        
        Usage:
            with client.session() as session:
                session.run("CREATE (n:Test)")
        """
        if not self.is_connected():
            self.connect()
        
        db = database or self.database
        session = self.driver.session(database=db)
        
        try:
            yield session
        finally:
            session.close()
    
    def execute_write(
        self, 
        query: str, 
        parameters: Optional[Dict] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute write query with transaction
        
        Args:
            query: Cypher query
            parameters: Query parameters
            database: Database name (optional)
            
        Returns:
            List of result records as dictionaries
        """
        if not self.is_connected():
            self.connect()
        
        parameters = parameters or {}
        
        def _execute_write_tx(tx: Transaction):
            result = tx.run(query, parameters)
            return [dict(record) for record in result]
        
        with self.session(database) as session:
            try:
                return session.execute_write(_execute_write_tx)
            except TransientError as e:
                logger.warning(f"Transient error, retrying: {e}")
                time.sleep(1)
                return session.execute_write(_execute_write_tx)
            except Exception as e:
                logger.error(f"Write query failed: {e}\nQuery: {query}")
                raise
    
    def execute_read(
        self,
        query: str,
        parameters: Optional[Dict] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute read query
        
        Args:
            query: Cypher query
            parameters: Query parameters
            database: Database name (optional)
            
        Returns:
            List of result records as dictionaries
        """
        if not self.is_connected():
            self.connect()
        
        parameters = parameters or {}
        
        def _execute_read_tx(tx: Transaction):
            result = tx.run(query, parameters)
            return [dict(record) for record in result]
        
        with self.session(database) as session:
            try:
                return session.execute_read(_execute_read_tx)
            except Exception as e:
                logger.error(f"Read query failed: {e}\nQuery: {query}")
                raise
    
    def execute_batch_write(
        self,
        queries: List[Tuple[str, Dict]],
        database: Optional[str] = None
    ) -> bool:
        """
        Execute multiple write queries in a single transaction
        
        Args:
            queries: List of (query, parameters) tuples
            database: Database name (optional)
            
        Returns:
            bool: True if successful
        """
        if not self.is_connected():
            self.connect()
        
        def _execute_batch_tx(tx: Transaction):
            for query, parameters in queries:
                tx.run(query, parameters or {})
        
        with self.session(database) as session:
            try:
                session.execute_write(_execute_batch_tx)
                logger.info(f"Executed batch of {len(queries)} queries")
                return True
            except Exception as e:
                logger.error(f"Batch write failed: {e}")
                raise
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get Neo4j server information"""
        try:
            result = self.execute_read("CALL dbms.components() YIELD name, versions, edition")
            if result:
                return {
                    "name": result[0].get("name"),
                    "version": result[0].get("versions", ["unknown"])[0],
                    "edition": result[0].get("edition")
                }
        except Exception as e:
            logger.error(f"Failed to get server info: {e}")
        
        return {"version": "unknown", "edition": "unknown"}
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get database statistics
        
        Returns:
            Dict with node counts, relationship counts, etc.
        """
        stats = {
            "total_nodes": 0,
            "total_relationships": 0,
            "node_labels": {},
            "relationship_types": {}
        }
        
        try:
            # Total nodes
            result = self.execute_read("MATCH (n) RETURN count(n) as count")
            stats["total_nodes"] = result[0]["count"] if result else 0
            
            # Total relationships
            result = self.execute_read("MATCH ()-[r]->() RETURN count(r) as count")
            stats["total_relationships"] = result[0]["count"] if result else 0
            
            # Node labels count
            result = self.execute_read("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(*) as count
                ORDER BY count DESC
            """)
            stats["node_labels"] = {r["label"]: r["count"] for r in result if r.get("label")}
            
            # Relationship types count
            result = self.execute_read("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(*) as count
                ORDER BY count DESC
            """)
            stats["relationship_types"] = {r["type"]: r["count"] for r in result if r.get("type")}
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
        
        return stats
    
    def clear_database(self, confirm: bool = False):
        """
        Clear all nodes and relationships - USE WITH CAUTION!
        
        Args:
            confirm: Must be True to execute (safety check)
        """
        if not confirm:
            raise ValueError("Must set confirm=True to clear database")
        
        logger.warning("⚠️  CLEARING ENTIRE NEO4J DATABASE")
        
        try:
            # Delete in batches to avoid memory issues
            batch_size = 10000
            
            while True:
                result = self.execute_write(f"""
                    MATCH (n)
                    WITH n LIMIT {batch_size}
                    DETACH DELETE n
                    RETURN count(n) as deleted
                """)
                
                deleted = result[0]["deleted"] if result else 0
                logger.info(f"Deleted {deleted} nodes")
                
                if deleted < batch_size:
                    break
            
            logger.warning("✅ Database cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            raise
    
    def create_constraints(self, constraints: List[str]):
        """
        Create database constraints
        
        Args:
            constraints: List of Cypher constraint statements
        """
        for constraint in constraints:
            try:
                self.execute_write(constraint)
                logger.info(f"Created constraint: {constraint[:50]}...")
            except Exception as e:
                # Constraint might already exist
                if "already exists" in str(e).lower():
                    logger.debug(f"Constraint already exists: {constraint[:50]}...")
                else:
                    logger.error(f"Failed to create constraint: {e}")
    
    def create_indexes(self, indexes: List[str]):
        """
        Create database indexes
        
        Args:
            indexes: List of Cypher index statements
        """
        for index in indexes:
            try:
                self.execute_write(index)
                logger.info(f"Created index: {index[:50]}...")
            except Exception as e:
                # Index might already exist
                if "already exists" in str(e).lower():
                    logger.debug(f"Index already exists: {index[:50]}...")
                else:
                    logger.error(f"Failed to create index: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.disconnect()


# Singleton instance for application-wide use
_neo4j_client_instance: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    """
    Get singleton Neo4j client instance
    
    Returns:
        Neo4jClient instance
    """
    global _neo4j_client_instance
    
    if _neo4j_client_instance is None:
        _neo4j_client_instance = Neo4jClient()
    
    return _neo4j_client_instance


def reset_neo4j_client():
    """Reset singleton instance (useful for testing)"""
    global _neo4j_client_instance
    
    if _neo4j_client_instance:
        _neo4j_client_instance.disconnect()
        _neo4j_client_instance = None