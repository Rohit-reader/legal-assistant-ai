import sys

def view_text_file(file_path: str, num_lines: int = 100):
    """View the first N lines of a text file with proper encoding."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                print(f"{i+1}: {line.strip()}")
    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python view_text.py <file_path> [num_lines]")
        return
    
    file_path = sys.argv[1]
    num_lines = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    print(f"Viewing first {num_lines} lines of {file_path}:")
    print("=" * 80)
    view_text_file(file_path, num_lines)

if __name__ == "__main__":
    main()
