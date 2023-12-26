import pandas as pd
from transformers import BertTokenizer

tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
data=pd.read_csv("dataset.tsv", sep="\t")

def tokenize():
    