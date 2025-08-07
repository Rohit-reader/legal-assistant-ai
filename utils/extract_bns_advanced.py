import fitz  # PyMuPDF
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

def extract_text_with_position(pdf_path: str) -> List[Dict]:
    """Extract text with position and style information."""
    doc = fitz.open(pdf_path)
    blocks = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES)['blocks']
        
        for block in text_blocks:
            if 'lines' in block:  # Text block
                for line in block['lines']:
                    for span in line['spans']:
                        text = span['text'].strip()
                        if text:  # Only process non-empty text
                            blocks.append({
                                'page': page_num + 1,
                                'text': text,
                                'font_size': span['size'],
                                'font': span['font'],
                                'bold': 'bold' in span['font'].lower(),
                                'x0': span['origin'][0],
                                'y0': span['origin'][1]
                            })
    return blocks

def identify_sections(blocks: List[Dict]) -> pd.DataFrame:
    """Identify sections from the extracted text blocks."""
    sections = []
    current_section = None
    current_content = []
    
    for block in blocks:
        text = block['text']
        
        # Look for section patterns
        section_match = re.match(r'^(\d+)\.\s+(.+?)(?=\s*$|\s+\d+\.)', text)
        
        if section_match and block['font_size'] > 10:  # Likely a section header
            # Save previous section if exists
            if current_section is not None and current_content:
                sections.append({
                    'section_number': current_section[0],
                    'section_title': current_section[1],
                    'content': ' '.join(current_content).strip(),
                    'page_number': current_section[2]
                })
            
            # Start new section
            section_num = section_match.group(1)
            section_title = section_match.group(2).strip()
            current_section = (section_num, section_title, block['page'])
            current_content = [text]
        elif current_section is not None:
            # Add content to current section
            current_content.append(text)
    
    # Add the last section
    if current_section is not None and current_content:
        sections.append({
            'section_number': current_section[0],
            'section_title': current_section[1],
            'content': ' '.join(current_content).strip(),
            'page_number': current_section[2]
        })
    
    return pd.DataFrame(sections)

def clean_sections(df: pd.DataFrame) -> pd.DataFrame:
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
    output_path = project_root / "data" / "raw" / "bns_sections_final.csv"
    
    print(f"Extracting BNS sections from {pdf_path}...")
    
    # Extract text with position information
    print("Extracting text blocks with position data...")
    blocks = extract_text_with_position(pdf_path)
    print(f"Extracted {len(blocks)} text blocks.")
    
    # Identify sections
    print("Identifying sections...")
    df = identify_sections(blocks)
    
    if not df.empty:
        # Clean and process the sections
        print("Cleaning and processing sections...")
        df = clean_sections(df)
        
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
