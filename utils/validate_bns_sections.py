import pandas as pd
from pathlib import Path
import json

def validate_bns_sections(input_path: str) -> dict:
    """Validate the cleaned BNS sections and prepare for integration."""
    # Read the cleaned sections
    df = pd.read_csv(input_path, encoding='utf-8')
    
    print(f"Validating {len(df)} BNS sections from {input_path}")
    
    # Basic validation
    validation_results = {
        'total_sections': len(df),
        'missing_titles': df['section_title'].isna().sum(),
        'missing_content': df['content'].isna().sum(),
        'duplicate_sections': df['section_number'].duplicated().sum(),
        'sections_by_length': {
            'short_titles': (df['section_title'].str.len() < 5).sum(),
            'short_content': (df['content'].str.len() < 20).sum()
        },
        'sample_sections': []
    }
    
    # Add sample sections for manual review
    sample_indices = [0, 10, 50, 100, 150, 200, 250, 300, -1]  # First, some middle, and last
    for idx in sample_indices:
        if abs(idx) < len(df):
            section = df.iloc[idx]
            validation_results['sample_sections'].append({
                'section_number': section['section_number'],
                'section_title': section['section_title'][:100] + ('...' if len(section['section_title']) > 100 else ''),
                'content_length': len(section['content']),
                'page_number': section['page_number']
            })
    
    return validation_results

def prepare_for_integration(df: pd.DataFrame, output_dir: str):
    """Prepare the BNS sections for integration with the legal assistant."""
    # Create a more structured format for the legal assistant
    structured_data = []
    
    for _, row in df.iterrows():
        structured_data.append({
            'section_number': row['section_number'],
            'section_title': row['section_title'],
            'content': row['content'],
            'metadata': {
                'page_number': row['page_number'],
                'source': 'Bharatiya Nyaya Sanhita, 2023',
                'extraction_method': 'automated_pdf_extraction'
            }
        })
    
    # Save as JSON for easy integration
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full dataset
    output_path = output_dir / 'bns_sections.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=2)
    
    # Save a simplified version for the model
    simplified_data = [{
        'section': f"{row['section_number']}. {row['section_title']}",
        'content': row['content']
    } for _, row in df.iterrows()]
    
    simplified_path = output_dir / 'bns_sections_simplified.json'
    with open(simplified_path, 'w', encoding='utf-8') as f:
        json.dump(simplified_data, f, ensure_ascii=False, indent=2)
    
    return {
        'full_dataset_path': str(output_path),
        'simplified_dataset_path': str(simplified_path),
        'total_sections': len(structured_data)
    }

def main():
    # Paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "processed" / "bns_sections_cleaned.csv"
    output_dir = project_root / "data" / "processed"
    
    # Validate the sections
    print("Validating BNS sections...")
    validation = validate_bns_sections(input_path)
    
    print("\nValidation Results:")
    print(f"- Total sections: {validation['total_sections']}")
    print(f"- Missing titles: {validation['missing_titles']}")
    print(f"- Missing content: {validation['missing_content']}")
    print(f"- Duplicate sections: {validation['duplicate_sections']}")
    print(f"- Sections with very short titles (<5 chars): {validation['sections_by_length']['short_titles']}")
    print(f"- Sections with very short content (<20 chars): {validation['sections_by_length']['short_content']}")
    
    print("\nSample sections:")
    for sample in validation['sample_sections']:
        print(f"\nSection {sample['section_number']} (Page {sample['page_number']}):")
        print(f"Title: {sample['section_title']}")
        print(f"Content length: {sample['content_length']} characters")
    
    # Prepare for integration
    print("\nPreparing BNS sections for integration...")
    df = pd.read_csv(input_path, encoding='utf-8')
    integration_results = prepare_for_integration(df, output_dir)
    
    print("\nIntegration Preparation Complete:")
    print(f"- Full dataset saved to: {integration_results['full_dataset_path']}")
    print(f"- Simplified dataset saved to: {integration_results['simplified_dataset_path']}")
    print(f"- Total sections processed: {integration_results['total_sections']}")

if __name__ == "__main__":
    main()
