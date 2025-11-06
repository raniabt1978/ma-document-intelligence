# core/vector_store.py
import chromadb
from chromadb.utils import embedding_functions
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, collection_name: str = "financial_documents", persist_directory: str = "./chroma_db"):
        """Initialize the vector store with ChromaDB"""
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Use sentence transformers for embeddings
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_document(self, 
                    text: str, 
                    metadata: Dict[str, Any],
                    doc_id: Optional[str] = None) -> Tuple[str, bool]:
        """
        Add a document to the vector store with deduplication
        Returns: (doc_id, is_new) tuple where is_new indicates if document was added
        """
        # Generate consistent doc_id based on filename if not provided
        if not doc_id:
            # Use filename as base for consistent ID
            filename = metadata.get('filename', '')
            if filename:
                doc_id = hashlib.md5(filename.encode()).hexdigest()
            else:
                # Fallback to content hash for consistency
                doc_id = hashlib.md5(text.encode()).hexdigest()
        
        # Check if document already exists by doc_id
        existing_chunks = self.get_document_chunks(doc_id)
        if existing_chunks:
            logger.info(f"Document {metadata.get('filename', doc_id)} already exists with same ID, skipping")
            return doc_id, False
        
        # Check for duplicate content by filename to handle re-uploads
        if metadata.get('filename'):
            existing_by_name = self.collection.get(
                where={"filename": metadata['filename']},
                limit=1
            )
            if existing_by_name['ids']:
                # Document with same filename exists - replace it
                logger.info(f"Document with filename {metadata['filename']} already exists, replacing")
                old_doc_id = existing_by_name['metadatas'][0].get('doc_id')
                if old_doc_id:
                    self.delete_document(old_doc_id)
        
        # Split into chunks with overlap for better context
        chunks = self._chunk_text(text)
        
        if not chunks:
            logger.warning(f"No chunks created for document {metadata.get('filename', doc_id)}")
            return doc_id, False
        
        # Prepare data for insertion
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks),
                "doc_id": doc_id,
                "chunk_size": len(chunk),
                "indexed_at": datetime.utcnow().isoformat()
            })
            
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(chunk_metadata)
        
        # Add to collection in batch
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Added document {metadata.get('filename', doc_id)} with {len(chunks)} chunks")
            return doc_id, True
        except Exception as e:
            logger.error(f"Error adding document to vector store: {e}")
            return doc_id, False
    
    def search(self, 
              query: str, 
              n_results: int = 5,
              filter_metadata: Optional[Dict] = None,
              deduplicate: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar documents with optional deduplication
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            deduplicate: Whether to remove duplicate/similar results
        
        Returns:
            List of search results with text, metadata, distance, and id
        """
        # Ensure we don't request more results than available
        total_chunks = self.collection.count()
        if total_chunks == 0:
            return []
        
        # Increase search size if deduplicating to ensure enough unique results
        search_n = min(n_results * 3 if deduplicate else n_results, total_chunks)
        
        kwargs = {
            "query_texts": [query],
            "n_results": search_n
        }
        
        if filter_metadata:
            kwargs["where"] = filter_metadata
        
        try:
            results = self.collection.query(**kwargs)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
        
        # Format and deduplicate results
        formatted_results = []
        seen_content = set()
        seen_doc_chunks = {}  # Track best chunk per document
        
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                text = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                result_id = results['ids'][0][i]
                
                doc_id = metadata.get('doc_id', '')
                
                if deduplicate:
                    # For each document, keep only the best (lowest distance) chunk
                    if doc_id in seen_doc_chunks:
                        if distance < seen_doc_chunks[doc_id]['distance']:
                            # Replace with better chunk
                            idx = next(j for j, r in enumerate(formatted_results) 
                                     if r['metadata'].get('doc_id') == doc_id)
                            formatted_results[idx] = {
                                'text': text,
                                'metadata': metadata,
                                'distance': distance,
                                'id': result_id
                            }
                            seen_doc_chunks[doc_id] = {'distance': distance, 'index': idx}
                    else:
                        # First chunk from this document
                        formatted_results.append({
                            'text': text,
                            'metadata': metadata,
                            'distance': distance,
                            'id': result_id
                        })
                        seen_doc_chunks[doc_id] = {
                            'distance': distance, 
                            'index': len(formatted_results) - 1
                        }
                else:
                    # No deduplication - add all results
                    formatted_results.append({
                        'text': text,
                        'metadata': metadata,
                        'distance': distance,
                        'id': result_id
                    })
                
                # Stop when we have enough results
                if deduplicate and len(formatted_results) >= n_results:
                    break
        
        # Sort by distance (best first) and limit to requested number
        formatted_results.sort(key=lambda x: x['distance'])
        return formatted_results[:n_results]
    
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a specific document"""
        try:
            results = self.collection.get(
                where={"doc_id": doc_id}
            )
        except Exception as e:
            logger.error(f"Error getting document chunks: {e}")
            return []
        
        chunks = []
        for i in range(len(results['ids'])):
            chunks.append({
                'id': results['ids'][i],
                'text': results['documents'][i],
                'metadata': results['metadatas'][i]
            })
        
        # Sort by chunk_id to maintain order
        chunks.sort(key=lambda x: x['metadata'].get('chunk_id', 0))
        return chunks
    
    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks of a document by doc_id"""
        try:
            results = self.collection.get(where={"doc_id": doc_id})
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {doc_id}")
                return len(results['ids'])
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
        return 0
    
    def delete_by_filename(self, filename: str) -> int:
        """Delete all chunks with a specific filename"""
        try:
            results = self.collection.get(where={"filename": filename})
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for filename {filename}")
                return len(results['ids'])
        except Exception as e:
            logger.error(f"Error deleting by filename: {e}")
        return 0
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks while preserving sentence boundaries
        
        Args:
            text: Text to split
            chunk_size: Target size for each chunk
            overlap: Number of characters to overlap between chunks
        
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # Split into sentences
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '.!?' and len(current) > 1:
                sentences.append(current.strip())
                current = ""
        
        if current.strip():
            sentences.append(current.strip())
        
        if not sentences:
            return [text] if text.strip() else []
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds chunk size and we have content
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(current_chunk)
                
                # Create overlap by including last few sentences
                overlap_start = max(0, i - 3)  # Include up to 3 previous sentences
                overlap_text = " ".join(sentences[overlap_start:i])
                
                # Start new chunk with overlap
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_size = len(current_chunk)
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_size += sentence_size + 1
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the vector store"""
        total_chunks = self.collection.count()
        
        # Get unique documents and files
        all_data = self.collection.get()
        unique_docs = {}
        unique_files = set()
        
        if all_data['metadatas']:
            for metadata in all_data['metadatas']:
                doc_id = metadata.get('doc_id')
                filename = metadata.get('filename')
                
                if doc_id and doc_id not in unique_docs:
                    unique_docs[doc_id] = {
                        'filename': filename,
                        'chunks': 0,
                        'doc_type': metadata.get('doc_type', 'unknown')
                    }
                
                if doc_id in unique_docs:
                    unique_docs[doc_id]['chunks'] += 1
                    
                if filename:
                    unique_files.add(filename)
        
        return {
            "total_chunks": total_chunks,
            "total_documents": len(unique_docs),
            "unique_files": len(unique_files),
            "collection_name": self.collection.name,
            "documents": list(unique_docs.values())
        }
    
    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List unique documents in the store with pagination
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
        
        Returns:
            List of unique documents with metadata
        """
        all_data = self.collection.get()
        docs_map = {}
        
        if all_data['metadatas']:
            for metadata in all_data['metadatas']:
                doc_id = metadata.get('doc_id')
                if doc_id and doc_id not in docs_map:
                    docs_map[doc_id] = {
                        'doc_id': doc_id,
                        'filename': metadata.get('filename', 'Unknown'),
                        'doc_type': metadata.get('doc_type', 'unknown'),
                        'upload_date': metadata.get('upload_date', ''),
                        'indexed_at': metadata.get('indexed_at', ''),
                        'file_size': metadata.get('file_size', 0),
                        'total_chunks': 0
                    }
                
                if doc_id in docs_map:
                    docs_map[doc_id]['total_chunks'] += 1
        
        # Convert to list and sort by indexed_at (newest first)
        doc_list = list(docs_map.values())
        doc_list.sort(key=lambda x: x.get('indexed_at', ''), reverse=True)
        
        # Apply pagination
        return doc_list[offset:offset + limit]
    
    def clear_all(self):
        """Clear all documents from the collection - use with caution!"""
        try:
            # Get collection name before deletion
            collection_name = self.collection.name
            
            # Delete the collection
            self.client.delete_collection(collection_name)
            
            # Recreate empty collection
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Cleared all documents from collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise