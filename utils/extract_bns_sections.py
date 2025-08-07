import os
import re
import PyPDF2
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

class BNSPDFExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.sections = []
        self.current_section = None
        self.current_content = []
        
    def extract_text_from_pdf(self) -> str:
        """Extract all text from the PDF file."""
        text = ""
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF file: {e}")
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize the extracted text."""
        # Replace multiple spaces and newlines with a single space
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and other common artifacts
        text = re.sub(r'\d+\s*\n', '', text)
        return text.strip()
    
    def is_section_header(self, line: str) -> bool:
        """Check if a line is a section header."""
        # Look for patterns like "1. Text" or "Section 1: Text"
        return bool(re.match(r'^(\d+[.)]|Section\s+\d+[:.)])\s+', line.strip()))
    
    def extract_sections(self) -> List[Dict[str, str]]:
        """Extract all sections from the PDF text."""
        text = self.extract_text_from_pdf()
        if not text:
            return []
        
        # Split text into lines and process each line
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        sections = []
        current_section = None
        current_content = []
        
        for line in lines:
            if self.is_section_header(line):
                # Save the previous section if it exists
                if current_section is not None and current_content:
                    sections.append({
                        'section_number': current_section[0],
                        'section_title': current_section[1],
                        'content': ' '.join(current_content).strip()
                    })
                
                # Start a new section
                match = re.match(r'^(\d+)[.)]\s*(.+)$', line)
                if not match:
                    match = re.match(r'^Section\s+(\d+)[:.)]\s*(.+)$', line)
                
                if match:
                    current_section = (match.group(1), match.group(2).strip())
                    current_content = []
            elif current_section is not None:
                # Add content to the current section
                current_content.append(line)
        
        # Add the last section
        if current_section is not None and current_content:
            sections.append({
                'section_number': current_section[0],
                'section_title': current_section[1],
                'content': ' '.join(current_content).strip()
            })
        
        return sections
    
    def save_to_csv(self, output_path: str) -> bool:
        """Save extracted sections to a CSV file."""
        try:
            sections = self.extract_sections()
            if not sections:
                print("No sections were extracted from the PDF.")
                return False
            
            df = pd.DataFrame(sections)
            df.to_csv(output_path, index=False)
            print(f"Successfully saved {len(sections)} sections to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            return False

def main():
    # Paths
    project_root = Path(__file__).parent.parent
    pdf_path = project_root / "250883_english_01042024.pdf"
    output_path = project_root / "data" / "raw" / "bns_sections_from_pdf.csv"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract and save sections
    extractor = BNSPDFExtractor(pdf_path)
    success = extractor.save_to_csv(output_path)
    
    if success:
        print("\nExtraction completed successfully!")
        print(f"Output file: {output_path}")
    else:
        print("\nExtraction failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
