import pandas as pd
import json

def change_json_to_jsonl(input_file, output_file):
    df = pd.read_json(input_file)

    # df.to_json(output_file, force_ascii=False)
    new = []
    for idx, row in df.iterrows():
        new_row = {
            "query": row['query'],
            "positive": row['pos'],
            "negative": row['neg'],
        }
        new.append(new_row)

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in new:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Converted {input_file} to {output_file} successfully.")

if __name__ == "__main__":
    input_file = '/home/fit02/dien-workspace/vlsp/data_drill/train-neg-top60-finetune.json'
    output_file = '/home/fit02/dien-workspace/vlsp/data_drill/train-neg-top60-finetune.jsonl'

    change_json_to_jsonl(input_file, output_file)