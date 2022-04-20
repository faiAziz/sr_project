import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from dataset import get_datasets, get_dataloaders

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

unique_labels = ["QUES", "EXCL", "PERD", "COMA", "HYPH", "APOS", "EMPT", "X"]
label2id = {k: v for v, k in enumerate(unique_labels)}
id2label = {v: k for v, k in enumerate(unique_labels)}

model = BertForTokenClassification.from_pretrained('bert-base-uncased',
                                                   num_labels=len(id2label),
                                                   id2label=id2label,
                                                   label2id=label2id)

model.to(device)

train_dataset, test_dataset = get_datasets()
train_loader, test_loader = get_dataloaders()

ids = train_dataset[0]["ids"].unsqueeze(0)
mask = train_dataset[0]["mask"].unsqueeze(0)
targets = train_dataset[0]["targets"].unsqueeze(0)
ids = ids.to(device)
mask = mask.to(device)
targets = targets.to(device)
outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
initial_loss = outputs[0]
print(initial_loss)

tr_logits = outputs[1]
print(tr_logits.shape)

