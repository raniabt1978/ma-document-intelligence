import requests
from bs4 import BeautifulSoup
from typing import Dict

class WebExtractor:
    def extract_text(self, url: str) -> str:
        """Extract text from a webpage"""
        try:
            # Get the webpage
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            return f"Error extracting from URL: {str(e)}"
    
    def extract_sec_filing(self, url: str) -> Dict:
        """Extract SEC filing with metadata"""
        text = self.extract_text(url)
        
        # Extract key sections if it's an SEC filing
        sections = {
            'full_text': text,
            'url': url
        }
        
        # Try to find common SEC sections
        if "FORM 10-K" in text or "FORM 10-Q" in text:
            sections['filing_type'] = '10-K' if "FORM 10-K" in text else '10-Q'
            
        return sections