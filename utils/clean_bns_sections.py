import pandas as pd
import re
from pathlib import Path

def clean_bns_sections(input_path: str, output_path: str) -> pd.DataFrame:
    """Clean and format the extracted BNS sections."""
    # Read the extracted sections
    df = pd.read_csv(input_path, encoding='utf-8')
    
    print(f"Loaded {len(df)} sections from {input_path}")
    
    # Clean section numbers - ensure they're numeric and properly formatted
    df['section_number'] = df['section_number'].astype(str).str.strip()
    
    # Remove any rows with non-numeric section numbers
    df = df[df['section_number'].str.match(r'^\d+$')]
    
    # Convert section numbers to integers for proper sorting
    df['section_num'] = df['section_number'].astype(int)
    df = df.sort_values('section_num')
    
    # Clean section titles
    df['section_title'] = df['section_title'].str.strip()
    
    # Remove any hash-like strings that might be artifacts
    df['section_title'] = df['section_title'].apply(
        lambda x: re.sub(r'[0-9a-f]{40,}', '', x).strip()
    )
    
    # Clean content
    df['content'] = df['content'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Drop temporary columns
    df = df[['section_number', 'section_title', 'content', 'page_number']]
    
    # Save cleaned data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\nCleaned {len(df)} BNS sections saved to {output_path}")
    print("\nSample of cleaned sections:")
    print(df[['section_number', 'section_title']].head(10).to_string(index=False))
    
    return df

def main():
    # Paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "raw" / "bns_sections_final.csv"
    output_path = project_root / "data" / "processed" / "bns_sections_cleaned.csv"
    
    # Clean the sections
    clean_bns_sections(input_path, output_path)

if __name__ == "__main__":
    main()
