from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score, classification_report
from dataset import load_prompts

from predict import predict_label

def evaluate_f1_scores(results, valid_labels):
    """
    Parses LLM predictions from a list of result dictionaries and calculates F1-scores.
    """
    y_true = []
    y_pred = []
    
    for row in results:
        ground_truth = row['ground_truth']
        raw_prediction = row['prediction']
        
        # 1. Safely Extract the Label
        # We check if any of the valid labels are contained within the generated text.
        # This prevents errors if the model outputs "The label is: Deflection."
        parsed_pred = "Unknown" # Fallback if the model hallucinated
        for label in valid_labels:
            if label.lower() in raw_prediction.lower():
                parsed_pred = label
                break
                
        y_true.append(ground_truth)
        y_pred.append(parsed_pred)

    # 2. Calculate F1 Scores
    # zero_division=0 prevents warnings if a class was never predicted
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=valid_labels, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', labels=valid_labels, zero_division=0)
    
    print(f"Macro F1-Score:    {macro_f1:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}\n")
    
    # 3. Generate the Classification Report
    # This builds a text report showing precision, recall, and F1 for each class.
    print("--- Detailed Classification Report ---")
    print(classification_report(y_true, y_pred, labels=valid_labels, zero_division=0))
    
    return macro_f1, weighted_f1


def evaluate_model_on_dataset(test_dataset, task_type, technique, model, tokenizer):
    """Iterates through the test dataset and collects predictions."""
    results = []
    task_mapping = {'Clear Reply': ['Explicit'],
                  'Ambivalent': ['Implicit', 'Dodging', 'General', 'Deflection', 'Partial/half-answer'],
                  'Clear Non-Reply': ['Declining to answer', 'Claims ignorance', 'Clarification']}
    
    for row in tqdm(test_dataset):
        question = row['interview_question']
        answer = row['interview_answer']
        clarity = row['clarity_label']
        ground_truth = None
        if task_type == 'task_1_clarity':
            ground_truth = clarity
        else:
            valid_evasions = task_mapping.get(clarity, [])
                
            # Gather the 3 annotations
            raw_votes = [row['annotator1'], row['annotator2'], row['annotator3']]
            
            # Filter step: only keep votes that exist in the valid mapping
            valid_votes = [vote for vote in raw_votes if vote in valid_evasions]
            
            if valid_votes == []:
                # Fallback for Zero Valid Votes: Pick the most common evasion tactic for that clarity class
                fallbacks = {   
                    'Clear Reply': 'Explicit',
                    'Ambivalent': 'Dodging', 
                    'Clear Non-Reply': 'Declining to answer'
                }
                evasion = fallbacks.get(clarity, None)
            else: 
                vote_counts = Counter(valid_votes)
                most_common = vote_counts.most_common()
                
                evasion = most_common[0][0]
                
            ground_truth = evasion

        prediction = predict_label(question, answer, task_type, technique, model, tokenizer)
        
        results.append({
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "prediction": prediction
        })
        
    return results