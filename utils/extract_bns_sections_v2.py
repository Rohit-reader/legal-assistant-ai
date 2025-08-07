import re
import PyPDF2
import pandas as pd
from pathlib import Path

def extract_sections_from_pdf(pdf_path: str) -> pd.DataFrame:
    """Extract sections from the BNS PDF file."""
    # Initialize variables
    sections = []
    current_section = None
    current_content = []
    
    # Regular expressions for section headers
    section_pattern = re.compile(r'^(\d+)[.)]\s+(.+?)(?=\d+[.)]|Section \d+[:.)]|$)')
    
    try:
        # Open the PDF file
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Process each page
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if not text:
                    continue
                
                # Clean up the text
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                text = re.sub(r'THE GAZETTE OF INDIA.*?\n', '', text, flags=re.DOTALL)
                
                # Split into lines and process
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                for line in lines:
                    # Check if this is a section header
                    match = section_pattern.match(line)
                    if match:
                        # Save previous section if exists
                        if current_section is not None and current_content:
                            sections.append({
                                'section_number': current_section[0],
                                'section_title': current_section[1],
                                'content': ' '.join(current_content),
                                'page_number': page_num
                            })
                        
                        # Start new section
                        current_section = (match.group(1), match.group(2))
                        current_content = [line]
                    elif current_section is not None:
                        # Add content to current section
                        current_content.append(line)
        
        # Add the last section
        if current_section is not None and current_content:
            sections.append({
                'section_number': current_section[0],
                'section_title': current_section[1],
                'content': ' '.join(current_content),
                'page_number': page_num
            })
        
        return pd.DataFrame(sections)
    
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return pd.DataFrame()

def main():
    # Paths
    project_root = Path(__file__).parent.parent
    pdf_path = project_root / "250883_english_01042024.pdf"
    output_path = project_root / "data" / "raw" / "bns_sections_improved.csv"
    
    print(f"Extracting sections from {pdf_path}...")
    df = extract_sections_from_pdf(pdf_path)
    
    if not df.empty:
        # Clean up the data
        df['section_title'] = df['section_title'].str.strip()
        df['content'] = df['content'].str.replace(r'\s+', ' ', regex=True)
        
        # Save to CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Successfully saved {len(df)} sections to {output_path}")
    else:
        print("No sections were extracted from the PDF.")

if __name__ == "__main__":
    main()
