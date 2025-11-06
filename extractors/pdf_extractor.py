import pdfplumber
from typing import List, Dict
import pandas as pd

class PDFExtractor:
    def extract_text(self, file) -> str:
        """Extract all text including tables"""
        text = ""
        
        with pdfplumber.open(file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract regular text
                page_text = page.extract_text() or ""
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text
                
                # Extract tables separately
                tables = page.extract_tables()
                if tables:
                    text += "\n[Tables found on this page:]\n"
                    for i, table in enumerate(tables):
                        text += f"\nTable {i+1}:\n"
                        # Convert table to readable format
                        df = pd.DataFrame(table[1:], columns=table[0])
                        text += df.to_string()
                        text += "\n"
                
                text += "\n\n"
        
        return text
    
    def extract_tables(self, file) -> List[pd.DataFrame]:
        """Extract just the tables as DataFrames"""
        all_tables = []
        
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if len(table) > 1:  # Has headers and data
                        df = pd.DataFrame(table[1:], columns=table[0])
                        all_tables.append(df)
        
        return all_tables