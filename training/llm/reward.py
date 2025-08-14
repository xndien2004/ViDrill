import re

def is_valid_format(response: str) -> bool:
    pattern = r'<answer>\[(\"[^\"]*\"(,\s*\"[^\"]*\")*)?\]</answer>'
    match = re.fullmatch(pattern, response.strip())
    return match is not None

def extract_ids_from_answer(text: str) -> list[str]:
    match = re.search(r"<answer>\[(.*?)\]</answer>", text.strip())
    if not match:
        return []
    id_list = match.group(1)
    ids = re.findall(r'["\'](.*?)["\']', id_list)
    return ids

def calculate_precision_recall(true_set, pred_set):
    correct = true_set & pred_set
    precision = len(correct) / len(pred_set) if pred_set else 0.0
    recall = len(correct) / len(true_set) if true_set else 0.0
    return precision, recall, correct

def calculate_f2(precision, recall):
    if precision + recall == 0:
        return 0.0
    return (5 * precision * recall) / (4 * precision + recall)

def combined_reward(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    scores = []

    for response, gold in zip(responses, answer):
        format_score = 0.1 if is_valid_format(response) else 0.0

        pred_ids = set(extract_ids_from_answer(response))
        gold_ids = set(extract_ids_from_answer(gold))
        # print(f"Gold response: {gold}")
        # print(f"Predicted IDs: {pred_ids}, Gold IDs: {gold_ids}")

        precision, recall, _ = calculate_precision_recall(gold_ids, pred_ids)
        f2 = calculate_f2(precision, recall)

        doc_score = f2 * 0.9
        total_score = format_score + doc_score
        print(f"Format score: {format_score}, Doc score: {doc_score}, Total score: {total_score}")
        scores.append(total_score)

    return scores
