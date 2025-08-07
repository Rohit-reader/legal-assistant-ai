import fitz  # PyMuPDF
import re
import pandas as pd
from pathlib import Path

def extract_complete_text(pdf_path: str) -> str:
    """Extract all text from the PDF as a single string."""
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        full_text += text + "\n"
    
    return full_text

def extract_sections_from_text(text: str) -> pd.DataFrame:
    """Extract sections from the complete text using pattern matching."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Pattern to match section numbers and titles
    # This pattern looks for numbers followed by a period and then text
    section_pattern = re.compile(r'\n(\d+)\.\s+([^.]+?)(?=\s+\d+\.|$)')
    
    # Find all matches
    matches = list(section_pattern.finditer(text))
    
    sections = []
    
    # Process each section
    for i in range(len(matches)):
        section_num = matches[i].group(1)
        section_title = matches[i].group(2).strip()
        
        # Get the content between this section and the next (or end of text)
        start_pos = matches[i].end()
        end_pos = matches[i+1].start() if i+1 < len(matches) else len(text)
        content = text[start_pos:end_pos].strip()
        
        # Add to sections list
        sections.append({
            'section_number': section_num,
            'section_title': section_title,
            'content': content
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
    output_path = project_root / "data" / "raw" / "bns_sections_simple.csv"
    
    print(f"Extracting text from {pdf_path}...")
    
    try:
        # Extract complete text
        text = extract_complete_text(pdf_path)
        
        # Save text to file for debugging
        text_path = project_root / "data" / "raw" / "bns_full_text.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Full text saved to {text_path}")
        
        # Extract sections from text
        print("Extracting sections from text...")
        df = extract_sections_from_text(text)
        
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
            print("No sections were extracted from the text.")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
