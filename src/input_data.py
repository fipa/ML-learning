from typing import List
import csv

def get_input_data(filepath: str) -> List[str]:
    new_feedback_examples = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    text = row[0].strip().strip('"')
                    new_feedback_examples.append(text)
        print(f"Loaded {len(new_feedback_examples)} examples.")
    except FileNotFoundError:
        print(f"Error: Examples file not found at {filepath}")
    except Exception as e:
        print(f"Error reading examples file: {e}")
    return new_feedback_examples