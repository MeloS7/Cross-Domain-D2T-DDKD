from openai import OpenAI
import argparse
import json
import yaml
import logging
import re
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def load_jsonl_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def pre_process_data(data, config, input_table):    
    processed_data = []
    for idx, item in enumerate(data):
        processed_data.append({
            'user_prompt': config['prompt_template'].format(data=input_table[idx]['formatted_input'], text=item['response']),
            'system_prompt': config['system_msg'],
            'original_response': item['response'],
        })
    return processed_data

def create_annotation(text, j, table_idx, metric_name, dataset_name, model_name):
    annotation_list = []
    current_pos = 0

    for error in j["errors"]:
        # # If the error text is "<OMISSION>", add an omission annotation
        # if error["text"] == "<OMISSION>":
        #     error["start"] = 0  # omission at the beginning of the text
        #     annotation_list.append(error)
        #     continue

        # Find the start index of the error in the text
        start_pos = text.lower().find(error["text"].lower(), current_pos)

        if current_pos != 0 and start_pos == -1:
            # Try from the beginning
            start_pos = text.find(error["text"])

        if start_pos == -1:
            logging.warning(f"Cannot find error {error} in text {text}, skipping")
            continue

        error["start"] = start_pos
        annotation_list.append(error)

        current_pos = start_pos + len(error["text"])

    annotation = {
        "annotator_id": metric_name,
        "dataset": dataset_name,
        "model": model_name,
        "table_idx": table_idx,
        "annotations": annotation_list,
    }

    return annotation

def evaluate_ent_desc(data, model, dataset_name, model_name):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    annotations = []

    for idx, item in tqdm(enumerate(data), total=len(data)):
        try:
            completion = client.chat.completions.create(  
                model=model,    
                messages=[
                    {
                        "role": "system",
                        "content": item['system_prompt'],
                    },
                    {
                        "role": "user",
                        "content": item['user_prompt'],
                    }
                ],
                temperature=0, # 0 for greedy search
                max_tokens=16384,
                seed=42
            )
        except Exception as e:
            logging.error(f"Error evaluating ent_desc: {e}")
            return {"errors": []}

        llm_judgment = completion.choices[0].message.content
        
        # Debug: print the raw response
        # print(f"Raw LLM judgment: '{llm_judgment}'")
        # print(f"Type: {type(llm_judgment)}")
        # print(f"Length: {len(llm_judgment) if llm_judgment else 0}")
        
        # Check if the response is empty or None
        if not llm_judgment or llm_judgment.strip() == "":
            logging.error("LLM returned empty response")
            # Create annotation indicating parsing failure
            llm_judgment_json = {
                "errors": [{
                    "reason": "LLM returned empty response - evaluation failed",
                    "text": "PARSING_FAILED",
                    "type": -1  # Special type to indicate parsing failure
                }]
            }
            logging.info("Using parsing failure annotation for empty LLM response")
        else:
            # Extract JSON from markdown code blocks if present
            json_content = llm_judgment.strip()
            if json_content.startswith("```json"):
                # Remove ```json from the beginning and ``` from the end
                json_content = json_content[7:]  # Remove ```json
                if json_content.endswith("```"):
                    json_content = json_content[:-3]  # Remove ```
                json_content = json_content.strip()
            elif json_content.startswith("```"):
                # Handle case where it's just ``` without json
                json_content = json_content[3:]
                if json_content.endswith("```"):
                    json_content = json_content[:-3]
                json_content = json_content.strip()
            
            # Convert the llm judgment to json
            try:
                llm_judgment_json = json.loads(json_content)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON: {e}")
                logging.error(f"Cleaned content: '{json_content}'")
                logging.error(f"Raw response: '{llm_judgment}'")
                # # Create annotation indicating parsing failure
                # llm_judgment_json = {
                #     "errors": [{
                #         "reason": f"JSON parsing failed: {str(e)} - evaluation failed",
                #         "text": "PARSING_FAILED",
                #         "type": -1  # Special type to indicate parsing failure
                #     }]
                # }
                # logging.info("Using parsing failure annotation for failed JSON parsing")

                # Attempt partial recovery: if the last error object is truncated,
                # extract all complete { ... } objects inside "errors": [ ... ].
                recovered_errors = []
                try:
                    errors_key_pos = json_content.find('"errors"')
                    if errors_key_pos != -1:
                        bracket_pos = json_content.find("[", errors_key_pos)
                        if bracket_pos != -1:
                            sub = json_content[bracket_pos + 1 :]
                            # Match top-level { ... } without nested braces; each should be one error object
                            for m in re.finditer(r"\{[^{}]*\}", sub, flags=re.DOTALL):
                                obj_str = m.group(0)
                                try:
                                    obj = json.loads(obj_str)
                                except json.JSONDecodeError:
                                    continue
                                if isinstance(obj, dict) and "text" in obj and "reason" in obj and "type" in obj:
                                    recovered_errors.append(obj)
                except Exception as e2:
                    logging.error(f"Partial recovery of errors failed: {e2}")

                if recovered_errors:
                    llm_judgment_json = {"errors": recovered_errors}
                    logging.info(
                        f"Recovered {len(recovered_errors)} error objects from partially truncated JSON."
                    )
                else:
                    # If recovery fails, fall back to PARSING_FAILED marker
                    llm_judgment_json = {
                        "errors": [
                            {
                                "reason": f"JSON parsing failed: {str(e)} - evaluation failed",
                                "text": "PARSING_FAILED",
                                "type": -1,  # Special type to indicate parsing failure
                            }
                        ]
                    }
                    logging.info("Using parsing failure annotation for failed JSON parsing")
        
        # Create annotation
        annotation = create_annotation(item['original_response'], llm_judgment_json, idx, model, dataset_name, model_name)
        annotations.append(annotation)

    return annotations

def save_annotations_jsonl(annotations, output_file):
    with open(output_file, 'w') as file:
        for annotation in annotations:
            json.dump(annotation, file, ensure_ascii=False)
            file.write('\n')

    logging.info(f"Saved annotations to {output_file}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('-o', '--output_file', type=str, required=False)
    parser.add_argument('--input_table', type=str, required=False, default='data/test/quintd/wikipedia/test_input_table.jsonl')
    parser.add_argument('-m', '--model', type=str, required=False, default="openai/gpt-4o")
    parser.add_argument('-c', '--config', type=str, required=False, default="eval/LLM_eval_ent_desc.yaml")
    parser.add_argument('-d', '--dataset_name', type=str, required=False, default="quintd_wikipedia_test")
    parser.add_argument('-n', '--model_name', type=str, required=False, default="gpt-4o")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    model = args.model
    config = args.config
    input_table = args.input_table
    dataset_name = args.dataset_name
    model_name = args.model_name
    
    # Load data
    data = load_jsonl_data(input_file)

    # Load input table
    input_table = load_jsonl_data(input_table)

    # Load config
    config = load_yaml(config)

    # Pre-process data
    data = pre_process_data(data, config, input_table)

    # Evaluate
    annotations = evaluate_ent_desc(data, model, dataset_name, model_name)

    # Save annotations
    save_annotations_jsonl(annotations, output_file)

if __name__ == "__main__":
    main()