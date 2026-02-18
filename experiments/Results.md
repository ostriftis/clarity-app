# Experiments on prompting techniques
Four different prompting techniques were evaluated before the final finetuning:
- **Default**: The model is prompted to classify the answer to the corresponding categories.
- **CoT**: The model is instructed to decide which category the answer belongs to, think step by step and not include the reasoning in the answer. Then it classifies the answer. 
- **Persona (role-playing)**: The model is prompted to play the role of a political analyst and then classify the answer. 
- **Definition aware**: The model is provided with definition for the classification labels. Then it is instructed to classify the answer. 

A balanced subset of the training set ($10\%$) was created and used to finetune small versions for the 2 tasks. Then, each model was evaluated on the test set. The results can be shown below: 

## Task 1 - Clarity


| Promting Technique    | F1-Score (weighted) | F1-Score (macro) |
| ----------- | ----------- | ----------- |
| Default            | 0.1407       | 0.1532      |
| Persona            | 0.1773        | 0.1732        |
| Chain-of-Thought   | 0.2408        | 0.2067        |
| Definition Aware   | 0.1037        | 0.1347      |


## Task 2 - Evasion


| Promting Technique    | F1-Score (weighted) | F1-Score (macro) |
| ----------- | ----------- | ----------- |
| Default            | 0.0655      | 0.0367       |
| Persona            | 0.0655        | 0.0367        |
| Chain-of-Thought   | 0.1395        | 0.0647        |
| Definition Aware   | 0.1669        | 0.0825       |


## Conclusion

The final model was fine-tuned with CoT for **Task 1: Clarity** and Definition-aware technique for **Task 2: Evasion**.

| Task | Base Model F1 | Fine-Tuned F1 | Improvement |
| :--- | :--- | :--- | :--- |
| **Clarity** | 0.0883 | **0.5187** | **+487%** |
| **Evasion** | 0.1275 | **0.3504** | **+174%** |

## President and Interview Date

Below we can see the effects of providing information about which president answered the question and when the interview was conducted:

| Task   |	None Provided |	Both Provided |
| ------ | --------- | --------- |
| Clarity |	0.5072 |	**0.5187** |
| Evasion |	 0.3326 |	**0.3504** |

It is apparent that providing these types of information improves classification f1-Score for both tasks.

## Reproducibility

The notebooks in this file were used to do these experiments. You can also run it locally by following the instructions on [README.md](README.md).