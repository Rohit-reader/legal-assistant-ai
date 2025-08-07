import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

def find_duplicate_sections(bns_data):
    """Find duplicate section numbers in the BNS dataset."""
    section_counts = defaultdict(list)
    
    for i, section in enumerate(bns_data):
        section_num = section['section_number']
        section_counts[section_num].append(i)
    
    # Find sections with duplicate numbers
    duplicates = {k: v for k, v in section_counts.items() if len(v) > 1}
    return duplicates

def fix_duplicate_sections(bns_data):
    """Fix duplicate section numbers by appending a letter suffix."""
    section_counts = {}
    fixed_data = []
    
    for section in bns_data:
        section_num = section['section_number']
        
        # Initialize or increment count for this section number
        if section_num in section_counts:
            section_counts[section_num] += 1
            # Append a letter suffix (a, b, c, ...) to the section number
            new_num = f"{section_num}{chr(96 + section_counts[section_num])}"  # 96 + 1 = 97 = 'a'
            
            # Create a copy of the section with the new number
            fixed_section = section.copy()
            fixed_section['section_number'] = new_num
            fixed_section['metadata']['original_section_number'] = section_num
            fixed_data.append(fixed_section)
            
            print(f"Fixed duplicate section {section_num} -> {new_num}: {section['section_title'][:50]}...")
        else:
            section_counts[section_num] = 1
            fixed_data.append(section)
    
    return fixed_data

def main():
    # Paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / "model" / "data" / "bns_sections.json"
    output_path = project_root / "model" / "data" / "bns_sections_fixed.json"
    
    # Load the BNS data
    with open(input_path, 'r', encoding='utf-8') as f:
        bns_data = json.load(f)
    
    print(f"Loaded {len(bns_data)} BNS sections")
    
    # Find duplicates
    duplicates = find_duplicate_sections(bns_data)
    
    if not duplicates:
        print("No duplicate section numbers found!")
        return
    
    print(f"\nFound {len(duplicates)} section numbers with duplicates:")
    for section_num, indices in duplicates.items():
        print(f"- Section {section_num}: {len(indices)} occurrences (indices: {indices})")
        for idx in indices[:3]:  # Show first 3 occurrences
            title = bns_data[idx]['section_title']
            print(f"  - {title[:80]}{'...' if len(title) > 80 else ''}")
        if len(indices) > 3:
            print(f"  - ... and {len(indices) - 3} more")
    
    # Fix duplicates
    print("\nFixing duplicate section numbers...")
    fixed_data = fix_duplicate_sections(bns_data)
    
    # Save fixed data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nFixed data saved to: {output_path}")
    print(f"Original sections: {len(bns_data)}")
    print(f"Fixed sections: {len(fixed_data)}")
    
    # Verify no duplicates in fixed data
    fixed_duplicates = find_duplicate_sections(fixed_data)
    if not fixed_duplicates:
        print("\nVerification: No duplicate section numbers remain!")
    else:
        print(f"\nWarning: {len(fixed_duplicates)} duplicate section numbers still exist!")
        for section_num, indices in list(fixed_duplicates.items())[:5]:
            print(f"- Section {section_num}: {len(indices)} occurrences")
    
    # Create a summary of changes
    summary = {
        'original_sections': len(bns_data),
        'fixed_sections': len(fixed_data),
        'duplicate_sections_found': len(duplicates),
        'duplicate_sections_after_fix': len(fixed_duplicates),
        'duplicate_details': {
            section_num: len(indices) 
            for section_num, indices in duplicates.items()
        }
    }
    
    # Save summary
    summary_path = project_root / "data" / "processed" / "bns_fix_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary of changes saved to: {summary_path}")

if __name__ == "__main__":
    main()
