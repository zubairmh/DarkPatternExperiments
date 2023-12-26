# Issues
# Dataset is not multilabel, single flag per data
# Small Dataset
# Text only

from transformers import BertForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd
complete=pd.read_csv("dataset.tsv", sep="\t")
dataset=complete[0:1000] # Train Split


labels = list(complete["category"].unique()) # All Labels
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}


print(labels) # Detection Flags


model_name="bert-base-uncased"
# model=BertForSequenceClassification.from_pretrained(model_name, num_labels=16, problem_type="multi_label_classification")
tokenizer=AutoTokenizer.from_pretrained(model_name)

print(tokenizer.tokenize(dataset["text"][0]))
# pipeline(model=model,tokenizer=tokenizer)