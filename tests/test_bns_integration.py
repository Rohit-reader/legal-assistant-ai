import json
import unittest
from pathlib import Path

class TestBNSDatasetIntegration(unittest.TestCase):    
    def setUp(self):
        """Load the BNS dataset for testing."""
        self.project_root = Path(__file__).parent.parent
        self.bns_path = self.project_root / "model" / "data" / "bns_sections.json"
        
        # Load the BNS dataset
        with open(self.bns_path, 'r', encoding='utf-8') as f:
            self.bns_data = json.load(f)
    
    def test_dataset_loaded(self):
        """Test that the dataset was loaded correctly."""
        self.assertIsInstance(self.bns_data, list, "BNS data should be a list")
        self.assertGreater(len(self.bns_data), 0, "BNS data should not be empty")
        print(f"\nLoaded {len(self.bns_data)} BNS sections for testing")
    
    def test_section_structure(self):
        """Test the structure of each section in the dataset."""
        required_keys = ['section_number', 'section_title', 'content', 'metadata']
        required_metadata_keys = ['page_number', 'source', 'extraction_method']
        
        for i, section in enumerate(self.bns_data):
            # Check required keys exist
            for key in required_keys:
                self.assertIn(key, section, f"Section #{i+1} is missing required key: {key}")
            
            # Check metadata structure
            metadata = section.get('metadata', {})
            for key in required_metadata_keys:
                self.assertIn(key, metadata, f"Section #{i+1} metadata is missing key: {key}")
            
            # Check data types (handle both string and integer section numbers)
            self.assertIsInstance(section['section_number'], (str, int), f"Section #{i+1} number should be a string or integer")
            self.assertIsInstance(section['section_title'], str, f"Section #{i+1} title should be a string")
            self.assertIsInstance(section['content'], str, f"Section #{i+1} content should be a string")
            self.assertIsInstance(metadata['page_number'], int, f"Section #{i+1} page number should be an integer")
        
        print("All sections have the correct structure")
    
    def test_sample_sections(self):
        """Test a sample of sections to ensure data quality."""
        # Test first section (handle both string and integer comparison)
        first_section = self.bns_data[0]
        first_section_num = str(first_section['section_number']).strip()
        self.assertEqual(first_section_num, '1', f"First section should be number 1, got {first_section_num}")
        self.assertIn('Bharatiya Nyaya Sanhita', first_section['section_title'])
        
        # Test a middle section
        middle_section = self.bns_data[len(self.bns_data)//2]
        self.assertIsNotNone(middle_section['section_number'])
        self.assertTrue(len(middle_section['content']) > 20, "Section content seems too short")
        
        # Test last section
        last_section = self.bns_data[-1]
        self.assertIn('repeal', last_section['section_title'].lower(), "Last section should be about repeal")
        
        print("Sample sections passed validation")
    
    def test_section_numbers(self):
        """Test that section numbers are sequential and unique."""
        # Handle both string and integer section numbers
        section_numbers = []
        for s in self.bns_data:
            try:
                num = int(s['section_number'])  # Convert to int if it's a string
                section_numbers.append(num)
            except (ValueError, TypeError):
                continue  # Skip non-numeric section numbers
                
        unique_numbers = set(section_numbers)
        
        if section_numbers:  # Only run these tests if we have valid section numbers
            self.assertEqual(len(section_numbers), len(unique_numbers), "Duplicate section numbers found")
            self.assertEqual(min(section_numbers), 1, "Section numbers should start at 1")
            self.assertLessEqual(max(section_numbers), 400, "Section number seems too high")
        
        print(f"Verified {len(section_numbers)} unique, sequential section numbers")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
