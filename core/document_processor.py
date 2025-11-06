# core/document_processor.py
import logging
from typing import Dict, Any, Optional
import io

from extractors.pdf_extractor import PDFExtractor
from extractors.excel_extractor import ExcelExtractor
from extractors.email_extractor import EmailExtractor
from extractors.word_extractor import WordExtractor
from extractors.ocr_extractor import OCRExtractor

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, enable_ocr: bool = True, ocr_threshold: int = 500):
        """
        Initialize document processor
        
        Args:
            enable_ocr: Whether to use OCR for scanned documents
            ocr_threshold: Minimum text length to consider document as text-based (not scanned)
        """
        self.pdf_extractor = PDFExtractor()
        self.excel_extractor = ExcelExtractor()
        self.email_extractor = EmailExtractor()
        self.word_extractor = WordExtractor()
        
        # OCR configuration
        self.enable_ocr = enable_ocr
        self.ocr_threshold = ocr_threshold
        
        if self.enable_ocr:
            try:
                self.ocr_extractor = OCRExtractor()
                # Check dependencies
                deps = self.ocr_extractor.check_dependencies()
                if not deps.get('tesseract'):
                    logger.warning("Tesseract not found. OCR will be disabled.")
                    self.enable_ocr = False
                if not deps.get('poppler'):
                    logger.warning("Poppler not found. PDF OCR may not work.")
            except Exception as e:
                logger.error(f"Failed to initialize OCR: {e}")
                self.enable_ocr = False
                self.ocr_extractor = None
        else:
            self.ocr_extractor = None
        
    def process(self, file, filename: str) -> Dict[str, Any]:
        """Process any supported document type"""
        try:
            file_ext = filename.lower().split('.')[-1]
            logger.info(f"Processing file: {filename} (type: {file_ext})")
            
            # Reset file position if possible
            if hasattr(file, 'seek'):
                file.seek(0)
            
            # Route to appropriate extractor
            if file_ext == 'pdf':
                result = self._process_pdf(file, filename)
            elif file_ext in ['xlsx', 'xls']:
                result = self._process_excel(file, filename)
            elif file_ext == 'eml':
                result = self._process_email(file, filename)
            elif file_ext in ['docx', 'doc']:
                result = self._process_word(file, filename)
            elif file_ext in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif']:
                result = self._process_image(file, filename)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # Ensure we have valid text
            if not result.get('text'):
                result['text'] = f"No text could be extracted from {filename}"
                
            # Add metadata
            result['filename'] = filename
            result['length'] = len(result['text'])
            
            logger.info(f"Processed {filename}: {result['length']} characters extracted")
            return result
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            return {
                'filename': filename,
                'doc_type': 'error',
                'text': f"Error processing document: {str(e)}",
                'length': 0,
                'error': str(e)
            }
    
    def _process_pdf(self, file, filename: str) -> Dict[str, Any]:
        """Process PDF with automatic OCR fallback"""
        # First try regular text extraction
        try:
            text = self.pdf_extractor.extract_text(file)
            
            # Reset file position for potential OCR
            if hasattr(file, 'seek'):
                file.seek(0)
            
            # Check if PDF might be scanned (very little text)
            text_length = len(text.strip())
            logger.info(f"PDF text extraction: {text_length} characters")
            
            if text_length < self.ocr_threshold and self.enable_ocr:
                logger.info(f"PDF has little text ({text_length} chars), attempting OCR...")
                
                try:
                    ocr_text = self.ocr_extractor.extract_text_from_pdf(file)
                    
                    # Use OCR if it extracted significantly more text
                    if len(ocr_text) > text_length + 100:
                        logger.info(f"OCR extracted {len(ocr_text)} characters, using OCR text")
                        return {
                            'doc_type': 'scanned_pdf',
                            'text': ocr_text
                        }
                    else:
                        logger.info("OCR didn't extract more text, using original")
                        
                except Exception as e:
                    logger.error(f"OCR failed: {e}")
                    # Continue with original text
            
            return {
                'doc_type': 'pdf',
                'text': text
            }
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            # Try OCR as last resort
            if self.enable_ocr:
                try:
                    if hasattr(file, 'seek'):
                        file.seek(0)
                    text = self.ocr_extractor.extract_text_from_pdf(file)
                    return {
                        'doc_type': 'scanned_pdf',
                        'text': text
                    }
                except:
                    pass
            
            raise
    
    def _process_excel(self, file, filename: str) -> Dict[str, Any]:
        """Process Excel file"""
        text = self.excel_extractor.extract_text(file)
        return {
            'doc_type': 'excel',
            'text': text
        }
    
    def _process_email(self, file, filename: str) -> Dict[str, Any]:
        """Process email file"""
        text = self.email_extractor.extract_text(file)
        return {
            'doc_type': 'email',
            'text': text
        }
    
    def _process_word(self, file, filename: str) -> Dict[str, Any]:
        """Process Word document"""
        text = self.word_extractor.extract_text(file)
        return {
            'doc_type': 'word',
            'text': text
        }
    
    def _process_image(self, file, filename: str) -> Dict[str, Any]:
        """Process image file with OCR"""
        if not self.enable_ocr:
            return {
                'doc_type': 'image',
                'text': 'OCR is disabled. Cannot extract text from images.'
            }
        
        text = self.ocr_extractor.extract_text_from_image(file)
        return {
            'doc_type': 'image',
            'text': text
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get current processor capabilities"""
        caps = {
            'supported_formats': {
                'pdf': True,
                'excel': True,
                'word': True,
                'email': True,
                'image': self.enable_ocr,
                'scanned_pdf': self.enable_ocr
            },
            'ocr_enabled': self.enable_ocr,
            'ocr_threshold': self.ocr_threshold
        }
        
        if self.enable_ocr and self.ocr_extractor:
            caps['ocr_dependencies'] = self.ocr_extractor.check_dependencies()
            
        return caps