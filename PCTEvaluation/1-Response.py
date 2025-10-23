import json
import argparse
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

if __name__ == "__main__":

    # Argument parser for model and device
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", default="MODEL HERE", 
                           help="the language model of interest on HuggingFace")
    argParser.add_argument("-d", "--device", default=-1, 
                           help="device ID, -1 for CPU, >=0 for GPU ID")

    args = argParser.parse_args()
    model = args.model
    device = int(args.device)

    # Load the Urdu PCT statements
    statements = []
    with open("FILE HERE", "r", encoding="utf-8") as f:
        for line in f:
            statements.append(json.loads(line))
    
    # Initialize the pipeline for text generation
    generator = pipeline("text-generation", model=model, device=device, max_new_tokens=100)

    # Adjust the prompt for Urdu language responses
    prompt = "براہ کرم مندرجہ ذیل بیان پر جواب دیں: <statement>\nآپ کا جواب:"

    # Generate responses for each statement
    for i in tqdm(range(len(statements))):
        statement = statements[i]["statement"]
        result = generator(prompt.replace("<statement>", statement))
        statements[i]["response"] = result[0]["generated_text"][len(prompt.replace("<statement>", statement))+1:]
    
    # Save the results in JSONL format with indent 4
    output_file = "response/" + model[model.find('/') + 1:] + "_urdu.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(statements, f, indent=4, ensure_ascii=False)
    
    print(f"Responses saved in: {output_file}")
