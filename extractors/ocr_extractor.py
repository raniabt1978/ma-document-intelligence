# extractors/ocr_extractor.py
import io
import os
import platform
import logging
from typing import Dict, List, Optional, Union
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import pytesseract
from pdf2image import convert_from_bytes, convert_from_path

logger = logging.getLogger(__name__)

class OCRExtractor:
    def __init__(self, tesseract_cmd: Optional[str] = None, lang: str = "eng"):
        """Initialize OCR extractor with automatic configuration"""
        # Configure Tesseract
        self._configure_tesseract(tesseract_cmd)
        self.lang = lang
        
        # Configure Poppler for PDF conversion
        self.poppler_path = self._find_poppler_path()
        
        # Log configuration
        try:
            logger.info(f"OCR Extractor initialized - Tesseract: {pytesseract.get_tesseract_version()}, Lang: {lang}")
        except:
            logger.info(f"OCR Extractor initialized - Lang: {lang}")
        
        if self.poppler_path:
            logger.info(f"Poppler path: {self.poppler_path}")

    def _configure_tesseract(self, tesseract_cmd: Optional[str] = None):
        """Configure Tesseract executable path"""
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        elif platform.system() == "Windows":
            # Common Windows paths
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\tesseract\tesseract.exe"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    logger.info(f"Found Tesseract at: {path}")
                    break
        
        # Verify Tesseract is available
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except Exception as e:
            logger.error(f"Tesseract not found: {e}")
            raise RuntimeError(
                "Tesseract not found. Please install Tesseract:\n"
                "- Mac: brew install tesseract\n"
                "- Ubuntu: sudo apt-get install tesseract-ocr\n"
                "- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
            )

    def _find_poppler_path(self) -> Optional[str]:
        """Find Poppler installation path"""
        if platform.system() != "Windows":
            return None  # Use system default on Unix
            
        # Common Windows Poppler paths
        possible_paths = [
            r"C:\Program Files\poppler-24.02.0\Library\bin",
            r"C:\Program Files\poppler-23.11.0\Library\bin",
            r"C:\Program Files\poppler\bin",
            r"C:\poppler\bin",
            r"C:\poppler-24.02.0\Library\bin",
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "poppler", "Library", "bin")
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "pdftoppm.exe")):
                logger.info(f"Found Poppler at: {path}")
                return path
        
        logger.warning("Poppler not found in common locations")
        return None

    def _preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """Enhanced preprocessing for better OCR accuracy"""
        try:
            # Convert to RGB if necessary
            if image.mode not in ('L', 'RGB'):
                image = image.convert('RGB')
            
            # Convert to grayscale
            image = image.convert('L')
            
            # Resize if too small (OCR works better on larger images)
            width, height = image.size
            if width < 1000 or height < 1000:
                scale = max(1000 / width, 1000 / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Auto contrast
            image = ImageOps.autocontrast(image)
            
            # Denoise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Sharpen
            image = image.filter(ImageFilter.SHARPEN)
            
            # Simple threshold without numpy (to avoid data type errors)
            threshold = 128  # Fixed threshold
            image = image.point(lambda p: 255 if p > threshold else 0)
            
            return image
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return image  # Return original if preprocessing fails

    def _deskew_if_needed(self, image: Image.Image) -> Image.Image:
        """Detect and correct image skew"""
        try:
            # Get orientation info
            osd_info = pytesseract.image_to_osd(image, config='--psm 0')
            
            # Parse rotation angle
            for line in osd_info.splitlines():
                if line.startswith("Rotate:"):
                    angle = int(line.split(":")[1].strip())
                    if angle != 0:
                        logger.info(f"Rotating image by {angle} degrees")
                        # Rotate with white background
                        image = image.rotate(-angle, expand=True, fillcolor=255)
                    break
                    
        except Exception as e:
            logger.debug(f"Could not detect orientation: {e}")
            # Continue without rotation
            
        return image

    def extract_text_from_image(self, image_file: Union[str, io.BytesIO, Image.Image]) -> str:
        """Extract text from image file"""
        try:
            # Handle different input types
            if isinstance(image_file, Image.Image):
                image = image_file
            elif isinstance(image_file, (str, io.BytesIO)):
                image = Image.open(image_file)
            elif hasattr(image_file, 'read'):  # File-like object
                image = Image.open(io.BytesIO(image_file.read()))
                if hasattr(image_file, 'seek'):
                    image_file.seek(0)
            else:
                return "Error: Unsupported image input type"
            
            # Try preprocessing (but don't fail if it doesn't work)
            try:
                image = self._deskew_if_needed(image)
                image = self._preprocess_for_ocr(image)
            except Exception as e:
                logger.warning(f"Preprocessing failed, using original image: {e}")
            
            # OCR with optimized config
            custom_config = r'--oem 3 --psm 6'
            
            try:
                text = pytesseract.image_to_string(
                    image, 
                    lang=self.lang,
                    config=custom_config,
                    timeout=30
                )
            except pytesseract.TesseractError as e:
                if "(-2," in str(e):
                    return "Error: OCR timeout - image too complex"
                else:
                    raise
            
            # Clean up text
            text = text.strip()
            
            # Check if we got meaningful text
            if not text or len(text) < 10:
                # Try with different PSM modes
                for psm in [3, 11, 12]:  # Different page segmentation modes
                    try:
                        alt_text = pytesseract.image_to_string(
                            image,
                            lang=self.lang,
                            config=f'--oem 3 --psm {psm}',
                            timeout=10
                        )
                        if len(alt_text) > len(text):
                            text = alt_text
                    except:
                        continue
            
            return text if text else "No text detected in image"
            
        except Exception as e:
            logger.error(f"Error in image OCR: {e}")
            return f"Error processing image: {str(e)}"

    def extract_text_from_pdf(self, pdf_file: Union[str, io.BytesIO]) -> str:
        """Extract text from PDF using OCR"""
        try:
            # Get PDF bytes
            if isinstance(pdf_file, str):
                with open(pdf_file, 'rb') as f:
                    pdf_bytes = f.read()
            elif hasattr(pdf_file, 'read'):
                pdf_bytes = pdf_file.read()
                if hasattr(pdf_file, 'seek'):
                    pdf_file.seek(0)
            else:
                return "Error: Invalid PDF input"
            
            # Check if PDF has content
            if not pdf_bytes or len(pdf_bytes) < 100:
                return "Error: PDF file is empty or too small"
            
            # Convert PDF to images
            try:
                # Configure conversion parameters
                convert_kwargs = {
                    'dpi': 200,  # Reduced for better performance
                    'fmt': 'png',
                    'grayscale': True,  # Convert to grayscale directly
                    'thread_count': 1,  # Single thread to avoid parallelism issues
                    'use_cropbox': True  # Use cropbox to remove margins
                }
                
                # Add poppler path for Windows
                if self.poppler_path:
                    convert_kwargs['poppler_path'] = self.poppler_path
                
                logger.info("Converting PDF to images...")
                images = convert_from_bytes(pdf_bytes, **convert_kwargs)
                
                if not images:
                    return "Error: Could not convert PDF to images"
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"PDF conversion error: {error_msg}")
                
                if "poppler" in error_msg.lower() or "pdftoppm" in error_msg.lower():
                    return (
                        "Error: Poppler not found. Please install:\n"
                        "- Mac: brew install poppler\n"
                        "- Ubuntu: sudo apt-get install poppler-utils\n"
                        "- Windows: Download from https://blog.alivate.com.au/poppler-windows/"
                    )
                else:
                    return f"Error converting PDF: {error_msg}"
            
            # Process each page
            logger.info(f"Processing {len(images)} pages with OCR...")
            pages_text = []
            
            for i, image in enumerate(images, start=1):
                try:
                    logger.info(f"Processing page {i}/{len(images)}")
                    
                    # Try preprocessing (but don't fail if it doesn't work)
                    try:
                        processed_image = self._preprocess_for_ocr(image)
                    except Exception as e:
                        logger.warning(f"Preprocessing failed for page {i}: {e}")
                        processed_image = image
                    
                    # OCR with timeout
                    try:
                        page_text = pytesseract.image_to_string(
                            processed_image,
                            lang=self.lang,
                            config='--oem 3 --psm 6',
                            timeout=30  # 30 second timeout per page
                        )
                        
                        if page_text.strip():
                            pages_text.append(f"--- Page {i} ---\n{page_text.strip()}")
                        else:
                            logger.warning(f"No text found on page {i}")
                            pages_text.append(f"--- Page {i} ---\n[No text detected]")
                            
                    except pytesseract.TesseractError as e:
                        if "(-2," in str(e):  # Timeout error
                            logger.warning(f"OCR timeout on page {i}")
                            pages_text.append(f"--- Page {i} ---\n[OCR timeout - page too complex]")
                        else:
                            logger.error(f"Tesseract error on page {i}: {e}")
                            pages_text.append(f"--- Page {i} ---\n[OCR error: {str(e)}]")
                        
                except Exception as e:
                    logger.error(f"Error processing page {i}: {e}")
                    pages_text.append(f"--- Page {i} ---\n[Processing error: {str(e)}]")
            
            # Combine all pages
            if pages_text:
                full_text = "\n\n".join(pages_text)
                logger.info(f"OCR completed. Extracted {len(full_text)} characters from {len(pages_text)} pages")
                return full_text
            else:
                return "No text could be extracted from the PDF"
                
        except Exception as e:
            logger.error(f"OCR PDF extraction error: {e}")
            return f"Error processing PDF: {str(e)}"

    def check_dependencies(self) -> Dict[str, bool]:
        """Check if all dependencies are properly installed"""
        status = {}
        
        # Check Tesseract
        try:
            version = pytesseract.get_tesseract_version()
            status['tesseract'] = True
            status['tesseract_version'] = str(version)
        except:
            status['tesseract'] = False
            
        # Check Poppler (Windows)
        if platform.system() == "Windows":
            status['poppler'] = bool(self.poppler_path)
        else:
            # Check if pdftoppm is in PATH
            import subprocess
            try:
                subprocess.run(['pdftoppm', '-v'], capture_output=True, check=False)
                status['poppler'] = True
            except:
                status['poppler'] = False
        
        # Check PIL
        try:
            from PIL import Image
            status['pillow'] = True
        except:
            status['pillow'] = False
            
        return status