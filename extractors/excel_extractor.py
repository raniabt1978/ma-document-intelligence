import pandas as pd
from typing import Dict, List

class ExcelExtractor:
    def extract_all_sheets(self, file) -> Dict[str, pd.DataFrame]:
        """Extract all sheets from Excel file"""
        excel_file = pd.ExcelFile(file)
        sheets_data = {}
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file, sheet_name=sheet_name)
            sheets_data[sheet_name] = df
            
        return sheets_data
    
    def extract_text(self, file) -> str:
        """Convert Excel to text format, preserving financial statement structure"""
        excel_file = pd.ExcelFile(file)
        text = ""
        
        for sheet_name in excel_file.sheet_names:
            text += f"\n--- Sheet: {sheet_name} ---\n"
            
            # Read the raw data to preserve structure
            df_raw = pd.read_excel(file, sheet_name=sheet_name, header=None)
            
            # Convert to string representation, keeping all data
            for idx, row in df_raw.iterrows():
                row_text = ""
                for col_val in row:
                    if pd.notna(col_val):
                        row_text += f"{str(col_val):<30}"  # Fixed width for alignment
                if row_text.strip():  # Only add non-empty rows
                    text += row_text + "\n"
            
            text += "\n\n"
            
        return text
    
    def extract_financial_data(self, file) -> Dict:
        """Extract structured financial data with row labels"""
        sheets_data = self.extract_all_sheets(file)
        financial_data = {}
        
        for sheet_name, df in sheets_data.items():
            # Try to identify the structure
            # Usually: First column = labels, then Notes, then Years
            if len(df.columns) >= 3:
                # Assume first column is row labels
                df_clean = df.copy()
                
                # Find columns that look like years
                year_columns = []
                for col in df.columns:
                    try:
                        year = float(str(col))
                        if 2000 <= year <= 2030:
                            year_columns.append(col)
                    except:
                        pass
                
                financial_data[sheet_name] = {
                    'full_data': df_clean,
                    'years': year_columns
                }
                
        return financial_data