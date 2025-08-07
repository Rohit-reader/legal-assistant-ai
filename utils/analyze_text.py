import re
import sys
from collections import defaultdict

def analyze_text_file(file_path: str):
    """Analyze the text file to identify potential section patterns."""
    try:
        # Read the file with proper encoding
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        print(f"Analyzing {file_path}...")
        print("=" * 80)
        
        # Look for potential section headers (lines with numbers followed by text)
        section_patterns = [
            r'^(\d+)\.\s+(.+?)(?=\n|$)',  # 1. Text
            r'^\(([a-z])\)\s+(.+)',  # (a) Text
            r'^Section\s+(\d+)[:.]?\s+(.+)',  # Section 1: Text
            r'^CHAPTER\s*([IVXLCDM]+)\s*\n(.+)',  # CHAPTER I\nTEXT
        ]
        
        # Try each pattern and count matches
        for i, pattern in enumerate(section_patterns, 1):
            matches = list(re.finditer(pattern, text, re.MULTILINE))
            print(f"\nPattern {i} ({pattern}): {len(matches)} matches")
            
            # Show first few matches
            for match in matches[:5]:
                print(f"  - {match.group(0)[:100]}")
        
        # Look for lines that might be section headers
        print("\nPotential section headers found:")
        potential_headers = re.findall(r'^\s*(\d+[.)]|Section\s+\d+|CHAPTER\s+[IVXLCDM]+|\s*[a-zA-Z\s]+\n)', 
                                     text, re.MULTILINE)
        
        # Print unique potential headers
        unique_headers = sorted(set(potential_headers))
        for header in unique_headers[:20]:  # Limit to first 20 unique headers
            print(f"  - {header.strip()}")
        
        # Count occurrences of "Section" followed by a number
        section_refs = re.findall(r'Section\s+\d+', text)
        print(f"\nFound {len(section_refs)} references to 'Section X' in the document.")
        
    except Exception as e:
        print(f"Error analyzing file: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_text.py <file_path>")
        return
    
    file_path = sys.argv[1]
    analyze_text_file(file_path)

if __name__ == "__main__":
    main()
