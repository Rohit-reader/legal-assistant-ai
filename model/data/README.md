# Legal Assistant - BNS Dataset

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
