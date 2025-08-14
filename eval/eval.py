import json

def calculate_precision_recall(true_set, pred_set):
    correct = true_set & pred_set
    precision = len(correct) / len(pred_set) if pred_set else 0.0
    recall = len(correct) / len(true_set) if true_set else 0.0
    return precision, recall, correct

def calculate_f2(precision, recall):
    if precision + recall == 0:
        return 0.0
    return (5 * precision * recall) / (4 * precision + recall)

def evaluate_predictions(true_path, pred_path):
    with open(true_path, 'r') as f:
        true_data = json.load(f)
    with open(pred_path, 'r') as f:
        pred_data = json.load(f)

    # Map QID to relevant law articles
    true_dict_full = {item.get("qid", item.get("id")): set(item["relevant_laws"]) for item in true_data}
    pred_dict_full = {item.get("qid", item.get("id")): set(item["relevant_laws"]) for item in pred_data}

    # Chỉ lấy giao của các qid có trong cả hai file
    common_qids = set(true_dict_full.keys()) & set(pred_dict_full.keys())
    true_dict = {qid: true_dict_full[qid] for qid in common_qids}
    pred_dict = {qid: pred_dict_full[qid] for qid in common_qids}

    qids = sorted(common_qids)
    precision_list = []
    recall_list = []
    error_cases = []
    detailed_log = []

    # Tạo dict ánh xạ từ qid sang predicted score nếu có
    pred_score_map = {
        item.get("qid", item.get("id")): item.get("scores", None)
        for item in pred_data
    }

    for qid in qids:
        true_set = true_dict[qid]
        pred_set = pred_dict[qid]

        precision, recall, correct = calculate_precision_recall(true_set, pred_set)

        precision_list.append(precision)
        recall_list.append(recall)

        if correct != true_set:
            error_cases.append(qid)
            detailed_log.append({
                "qid": qid,
                "true": sorted(true_set),
                "pred": sorted(pred_set),
                "missed": sorted(true_set - pred_set),
                "wrong": sorted(pred_set - true_set),
                "precision": precision,
                "recall": recall,
                "score_predict": pred_score_map.get(qid)
            })

    avg_precision = sum(precision_list) / len(precision_list)
    avg_recall = sum(recall_list) / len(recall_list)
    f2_score = calculate_f2(avg_precision, avg_recall)

    print(f"\nEvaluation Metrics:")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f}")
    print(f"F2-score:  {f2_score:.4f}")

    print(f"\nTotal incorrect examples: {len(error_cases)} / {len(qids)}")
    if error_cases:
        print(f"Wrong QIDs: {error_cases}")

    with open("detailed_analysis.json", "w") as f:
        json.dump(detailed_log, f, ensure_ascii=False, indent=4)

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f2": f2_score,
        "wrong_qids": error_cases,
        "detailed_analysis": detailed_log
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate predictions against true labels.")
    parser.add_argument("--true_path", type=str, help="Path to the true labels JSON file.")
    parser.add_argument("--pred_path", type=str, help="Path to the predicted labels JSON file.")

    args = parser.parse_args()

    evaluate_predictions(args.true_path, args.pred_path)
