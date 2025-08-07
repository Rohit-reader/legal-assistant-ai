import fitz  # PyMuPDF
import re
import pandas as pd
from pathlib import Path

def extract_sections_with_fitz(pdf_path: str) -> pd.DataFrame:
    """Extract BNS sections using PyMuPDF's layout analysis."""
    doc = fitz.open(pdf_path)
    sections = []
    current_section = None
    current_content = []
    
    # Regular expression to match section headers (e.g., "1. Text" or "Section 1: Text")
    section_pattern = re.compile(r'^(\d+)[.)]\s+(.+?)(?=\d+[.)]|Section\s+\d+[:.)]|$)')
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        
        # Skip pages that don't contain section content
        if not text.strip() or "THE GAZETTE OF INDIA" in text:
            continue
            
        # Clean up the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into lines and process
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

def main():
    # Paths
    project_root = Path(__file__).parent.parent
    pdf_path = project_root / "250883_english_01042024.pdf"
    output_path = project_root / "data" / "raw" / "bns_sections_final.csv"
    
    print(f"Extracting sections from {pdf_path} using PyMuPDF...")
    df = extract_sections_with_fitz(pdf_path)
    
    if not df.empty:
        # Clean up the data
        df = df[~df['section_number'].str.contains('[a-zA-Z]', na=False)]  # Remove non-numeric sections
        df = df.drop_duplicates(subset=['section_number'], keep='first')  # Keep first occurrence of each section
        
        # Save to CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Successfully saved {len(df)} sections to {output_path}")
        
        # Print sample of extracted sections
        print("\nSample of extracted sections:")
        print(df[['section_number', 'section_title']].head(10).to_string(index=False))
    else:
        print("No sections were extracted from the PDF.")

if __name__ == "__main__":
    main()
