import json 
import ast

def main(args):
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert string representations of lists to actual lists
    # min_length = []
    # for item in data:
    #     if isinstance(item['pos'], str):
    #         item['pos'] = ast.literal_eval(item['pos'])
    #     if isinstance(item['neg'], str):
    #         item['neg'] = ast.literal_eval(item['neg'])
    #     min_length.append(len(item['neg']))

    # print(f"min length of 'neg' lists: {min(min_length)}")

    # Slice the 'neg' list for each item
    for item in data:
        item['neg'] = item['neg'][args.start:args.end]
        item['neg_scores'] = item['neg_scores'][args.start:args.end]

    # Save the modified data back to a JSON file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Process JSON data to slice 'neg' lists.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output JSON file.')
    parser.add_argument('--start', type=int, default=0, help='Start index for slicing the neg list.')
    parser.add_argument('--end', type=int, default=60, help='End index for slicing the neg list.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
