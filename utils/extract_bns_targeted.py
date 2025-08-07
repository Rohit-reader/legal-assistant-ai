import fitz  # PyMuPDF
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

def extract_bns_sections(pdf_path: str) -> pd.DataFrame:
    """Extract BNS sections using targeted pattern matching based on PDF structure."""
    doc = fitz.open(pdf_path)
    sections = []
    current_section = None
    current_content = []
    
    # Regular expression to match section numbers and titles
    section_pattern = re.compile(r'^(\d+)\.\s+(.+?)(?=\n|$)')
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        
        # Skip pages that are just headers/footers
        if not text.strip() or "THE GAZETTE OF INDIA" in text:
            continue
            
        # Split into lines and process each line
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for line in lines:
            # Check if this is a section header
            match = section_pattern.search(line)
            if match:
                # Save previous section if exists
                if current_section is not None and current_content:
                    sections.append({
                        'section_number': current_section[0],
                        'section_title': current_section[1],
                        'content': ' '.join(current_content).strip(),
                        'page_number': page_num + 1
                    })
                
                # Start new section
                section_num = match.group(1)
                section_title = match.group(2).strip()
                current_section = (section_num, section_title)
                current_content = [line]
            elif current_section is not None:
                # Add content to current section
                current_content.append(line)
    
    # Add the last section
    if current_section is not None and current_content:
        sections.append({
            'section_number': current_section[0],
            'section_title': current_section[1],
            'content': ' '.join(current_content).strip(),
            'page_number': page_num + 1
        })
    
    return pd.DataFrame(sections)

def clean_extracted_sections(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and process the extracted sections."""
    if df.empty:
        return df
    
    # Clean section numbers and titles
    df['section_number'] = df['section_number'].astype(str).str.strip('.')
    df['section_title'] = df['section_title'].str.strip()
    
    # Remove any non-numeric section numbers
    df = df[df['section_number'].str.match(r'^\d+$')]
    
    # Convert section numbers to integers for proper sorting
    df['section_num'] = df['section_number'].astype(int)
    df = df.sort_values('section_num')
    df = df.drop('section_num', axis=1)
    
    # Clean up content
    df['content'] = df['content'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return df

def main():
    # Paths
    project_root = Path(__file__).parent.parent
    pdf_path = project_root / "250883_english_01042024.pdf"
    output_path = project_root / "data" / "raw" / "bns_sections_extracted.csv"
    
    print(f"Extracting BNS sections from {pdf_path}...")
    df = extract_bns_sections(pdf_path)
    
    if not df.empty:
        # Clean and process the sections
        df = clean_extracted_sections(df)
        
        # Save to CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"\nSuccessfully extracted {len(df)} BNS sections to {output_path}")
        print("\nSample of extracted sections:")
        print(df[['section_number', 'section_title']].head(10).to_string(index=False))
    else:
        print("No sections were extracted from the PDF.")

if __name__ == "__main__":
    main()
