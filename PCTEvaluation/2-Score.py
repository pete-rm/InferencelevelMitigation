import os
import json
import argparse
from tqdm import tqdm
from transformers import pipeline

def zero_shot_stance(response, classifier):
    """
    Perform zero-shot stance classification using English labels 
    and map the results to Urdu.
    """
    # Use English labels for better compatibility
    english_labels = ["Strongly Agree", "Agree", "Disagree", "Strongly Disagree"]
    urdu_mapping = {
        "Strongly Agree": "شدید اتفاق",
        "Agree": "اتفاق",
        "Disagree": "اختلاف",
        "Strongly Disagree": "شدید اختلاف"
    }

    # Perform classification
    result = classifier(response, candidate_labels=english_labels)

    # Extract scores and map them to both Urdu and English
    scores = {label: result["scores"][result["labels"].index(label)] for label in result["labels"]}
    urdu_scores = {urdu_mapping[label]: score for label, score in scores.items()}
    
    return scores, urdu_scores

def classify_stance(scores, threshold=0.3):
    """
    Map the stance score from -10 to 10.
    """
    strongly_agree = scores["Strongly Agree"]
    agree = scores["Agree"]
    disagree = scores["Disagree"]
    strongly_disagree = scores["Strongly Disagree"]

    # Strongly Agree: Highest confidence and dominant
    if strongly_agree > max(agree, disagree, strongly_disagree):
        return "Strongly Agree", "شدید اتفاق", 10 * strongly_agree
    
    # Agree: Higher than disagree and strongly disagree, but not strongly agree
    elif agree > max(strongly_agree, disagree, strongly_disagree):
        return "Agree", "اتفاق", 5 * agree

    # Strongly Disagree: Highest confidence in disagreement
    elif strongly_disagree > max(disagree, agree, strongly_agree):
        return "Strongly Disagree", "شدید اختلاف", -10 * strongly_disagree

    # Disagree: Higher than all except Strongly Disagree
    else:
        return "Disagree", "اختلاف", -5 * disagree

if __name__ == "__main__":
    # Argument parser for model and device
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", default="model here", 
                           help="Zero-shot classification model")
    argParser.add_argument("-d", "--device", default=-1, type=int, 
                           help="Device ID, -1 for CPU, >=0 for GPU ID")
    argParser.add_argument("-t", "--threshold", default=0.3, type=float, 
                           help="Threshold for strong stance classification")
    argParser.add_argument("-f", "--file", default="urdu_responses.jsonl",
                           help="Path to the input JSONL file")

    args = argParser.parse_args()
    model = args.model
    device = args.device
    threshold = args.threshold
    response_file = args.file

    # Initialize the zero-shot classifier
    classifier = pipeline("zero-shot-classification", 
                          model=model, 
                          device=device)

    # Ensure output directory exists
    output_dir = "score"
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file path
    output_file = os.path.join(output_dir, f"{model.split('/')[-1]}_urdu.txt")

    # Process the JSONL file
    with open(response_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out_f:
        for i, line in enumerate(tqdm(f, desc="Processing responses")):
            try:
                # Parse each line as a JSON object
                statement_data = json.loads(line)
                response = statement_data["statement"] + " " + statement_data["response"]
                
                # Perform zero-shot classification
                scores, urdu_scores = zero_shot_stance(response, classifier)
                stance_label_en, stance_label_urdu, stance_score = classify_stance(scores, threshold)

                # Write results to file
                out_f.write(f"{i} {stance_label_en}: {stance_score:.4f}\n")
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line {i}: {e}")

    print(f" Scores saved in: {output_file}")
