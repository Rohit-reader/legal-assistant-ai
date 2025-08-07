import json
import shutil
from pathlib import Path

def integrate_bns_dataset():
    """Integrate the BNS dataset into the legal assistant application."""
    project_root = Path(__file__).parent.parent
    
    # Paths
    bns_json_path = project_root / "data" / "processed" / "bns_sections.json"
    model_data_dir = project_root / "model" / "data"
    
    # Ensure model data directory exists
    model_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the BNS JSON file to the model data directory
    target_path = model_data_dir / "bns_sections.json"
    shutil.copy2(bns_json_path, target_path)
    
    print(f"BNS dataset copied to: {target_path}")
    
    # Update the model's data loading code if needed
    # (This would depend on how your model loads its data)
    
    # Create or update a README file with dataset information
    readme_path = model_data_dir / "README.md"
    readme_content = """# Legal Assistant - BNS Dataset

This directory contains the Bharatiya Nyaya Sanhita (BNS) 2023 dataset used by the Legal Assistant AI.

## Files

- `bns_sections.json`: Complete BNS sections with metadata

## Dataset Information

- **Source**: The Gazette of India - Bharatiya Nyaya Sanhita, 2023
- **Extraction Date**: 2024-08-08
- **Total Sections**: 359
- **Format**: JSON array of section objects

## Schema

Each section in the JSON file has the following structure:

```json
{
  "section_number": "1",
  "section_title": "Short Title",
  "content": "Full text of the section...",
  "metadata": {
    "page_number": 1,
    "source": "Bharatiya Nyaya Sanhita, 2023",
    "extraction_method": "automated_pdf_extraction"
  }
}
```

## Usage

Load the dataset in Python:

```python
import json

with open('bns_sections.json', 'r', encoding='utf-8') as f:
    bns_sections = json.load(f)
```
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"Dataset documentation updated: {readme_path}")
    
    # Verify the dataset can be loaded
    try:
        with open(target_path, 'r', encoding='utf-8') as f:
            bns_data = json.load(f)
        print(f"Successfully loaded {len(bns_data)} BNS sections.")
        return True
    except Exception as e:
        print(f"Error loading BNS dataset: {e}")
        return False

def main():
    print("Integrating BNS dataset into the legal assistant application...")
    success = integrate_bns_dataset()
    
    if success:
        print("\nIntegration successful! The BNS dataset is now ready to be used by the legal assistant.")
        print("\nNext steps:")
        print("1. Update the model training script to use the new dataset if needed")
        print("2. Test the legal assistant with the updated BNS sections")
        print("3. Deploy the updated application with the enhanced dataset")
    else:
        print("\nIntegration failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
