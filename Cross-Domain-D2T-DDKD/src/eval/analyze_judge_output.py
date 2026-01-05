import argparse
import json
import matplotlib.pyplot as plt

ERROR_TYPES = {
    0: "Incorrect fact",
    1: "Not checkable",
    2: "Misleading",
    3: "Other",
}

def load_jsonl_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def analyze_data(data):
    correct_count = 0
    # Table 3 type: Total count of each error type
    error_types_count = {error_type: 0 for error_type in ERROR_TYPES}
    # Table 4 type: Count of outputs containing at least one error of each type
    outputs_with_error_type = {error_type: 0 for error_type in ERROR_TYPES}
    
    for item in data:
        # If there are no annotations, it is correct
        if len(item['annotations']) == 0:
            correct_count += 1
        else:
            # Track which error types appear in this output
            error_types_in_output = set()
            
            for error in item['annotations']:
                error_type = error['type']
                error_types_count[error_type] += 1
                error_types_in_output.add(error_type)
            
            # Count outputs containing each error type
            for error_type in error_types_in_output:
                outputs_with_error_type[error_type] += 1

    total_count = len(data)

    return error_types_count, outputs_with_error_type, correct_count, total_count

def visualize_data(error_types_count, outputs_with_error_type, correct_count, total_count):
    print("=" * 80)
    print("Table 3 Style: Average numbers of errors per output")
    print("=" * 80)
    print(f"{'Error Type':<20} {'Total Count':<15} {'Average per Output':<20}")
    print("-" * 55)
    
    total_errors = sum(error_types_count.values())
    all_categories_total = total_errors
    
    for error_type, count in error_types_count.items():
        avg_per_output = count / total_count
        print(f"{ERROR_TYPES[error_type]:<20} {count:<15} {avg_per_output:<20.3f}")
    
    print("-" * 55)
    print(f"{'All categories':<20} {all_categories_total:<15} {all_categories_total/total_count:<20.3f}")
    print(f"{'# Tokens':<20} {'N/A':<15} {'N/A':<20}")
    
    print("\n" + "=" * 80)
    print("Table 4 Style: Percentage of outputs containing at least one error")
    print("=" * 80)
    print(f"{'Error Type':<20} {'Outputs with Error':<20} {'Percentage':<15}")
    print("-" * 55)
    
    outputs_with_any_error = total_count - correct_count
    
    for error_type, count in outputs_with_error_type.items():
        percentage = (count / total_count) * 100
        print(f"{ERROR_TYPES[error_type]:<20} {count:<20} {percentage:<15.1f}%")
    
    print("-" * 55)
    print(f"{'All categories':<20} {outputs_with_any_error:<20} {(outputs_with_any_error/total_count)*100:<15.1f}%")
    
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Total outputs: {total_count}")
    print(f"Correct outputs: {correct_count}")
    print(f"Outputs with errors: {outputs_with_any_error}")
    print(f"Accuracy: {correct_count / total_count * 100:.2f}%")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True)
    args = parser.parse_args()

    input_file = args.input_file
    
    # Load data
    data = load_jsonl_data(input_file)

    # Analyze data
    error_types_count, outputs_with_error_type, correct_count, total_count = analyze_data(data)
    
    # Visualize data
    visualize_data(error_types_count, outputs_with_error_type, correct_count, total_count)


if __name__ == "__main__":
    main()