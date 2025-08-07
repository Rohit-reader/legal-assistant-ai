import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path

def extract_pdf_toc(pdf_path: str) -> pd.DataFrame:
    """Extract the table of contents from the PDF."""
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    
    # Convert TOC to DataFrame
    df = pd.DataFrame(toc, columns=['level', 'title', 'page', 'other'])
    return df

def main():
    # Paths
    project_root = Path(__file__).parent.parent
    pdf_path = project_root / "250883_english_01042024.pdf"
    output_path = project_root / "data" / "raw" / "bns_toc.csv"
    
    print(f"Extracting table of contents from {pdf_path}...")
    
    try:
        # Extract TOC
        toc_df = extract_pdf_toc(pdf_path)
        
        if not toc_df.empty:
            # Save to CSV
            output_path.parent.mkdir(parents=True, exist_ok=True)
            toc_df.to_csv(output_path, index=False)
            
            print(f"\nSuccessfully extracted TOC with {len(toc_df)} entries to {output_path}")
            print("\nSample of TOC entries:")
            print(toc_df.head(10).to_string(index=False))
        else:
            print("No table of contents found in the PDF.")
    except Exception as e:
        print(f"Error extracting TOC: {e}")

if __name__ == "__main__":
    main()
