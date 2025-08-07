import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path

def analyze_pdf_structure(pdf_path: str, output_csv: str = None):
    """Analyze the PDF structure and extract text with coordinates."""
    doc = fitz.open(pdf_path)
    data = []
    
    for page_num in range(min(10, len(doc))):  # Analyze first 10 pages
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        
        for b in blocks:
            if "lines" in b:  # Text block
                for line in b["lines"]:
                    for span in line["spans"]:
                        data.append({
                            "page": page_num + 1,
                            "text": span["text"].strip(),
                            "size": span["size"],
                            "font": span["font"],
                            "bold": "bold" in span["font"].lower(),
                            "x0": span["origin"][0],
                            "y0": span["origin"][1]
                        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV if output path is provided
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"PDF structure analysis saved to {output_csv}")
    
    return df

def main():
    # Paths
    project_root = Path(__file__).parent.parent
    pdf_path = project_root / "250883_english_01042024.pdf"
    output_csv = project_root / "data" / "raw" / "pdf_structure_analysis.csv"
    
    print(f"Analyzing PDF structure: {pdf_path}")
    df = analyze_pdf_structure(pdf_path, output_csv)
    
    # Print summary of the analysis
    if not df.empty:
        print("\nPDF Structure Summary:")
        print(f"Total pages analyzed: {df['page'].nunique()}")
        print(f"Total text blocks: {len(df)}")
        print("\nSample of text blocks:")
        print(df.head(20)[['page', 'text', 'size', 'bold']].to_string(index=False))
    else:
        print("No text blocks were found in the PDF.")

if __name__ == "__main__":
    main()
