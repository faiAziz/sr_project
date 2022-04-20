import pickle
import torch
from torch.utils.data import Dataset, DataLoader


class dataset(Dataset):
    def __init__(self, prefix):
        with open("data/" + prefix + "_data.pkl", "rb") as f:
            data = pickle.load(f)
        self.dataset = data
        self.text_dataset = data["text"]
        self.label_dataset = data["label"]
        self.attn_masks = data["mask"]
        self.len = len(self.text_dataset)
        unique_labels = ["QUES", "EXCL", "PERD", "COMA", "HYPH", "APOS", "EMPT", "X"]
        self.label2id = {k: v for v, k in enumerate(unique_labels)}
        self.id2label = {v: k for v, k in enumerate(unique_labels)}

    def __getitem__(self, index):
        sentence = self.text_dataset[index]
        labels = self.label_dataset[index]
        attn_mask = self.attn_masks[index]
        label_ids = [self.label2id[label] for label in labels]

        return {
            'ids': torch.tensor(sentence, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            'targets': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return self.len


def get_datasets():
    train_dataset = dataset("train")
    test_dataset = dataset("test")

    return train_dataset, test_dataset


def get_dataloaders():
    TRAIN_BATCH_SIZE = 4
    TEST_BATCH_SIZE = 2
    train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    test_params = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    train_dataset, test_dataset = get_datasets()

    train_loader = DataLoader(train_dataset, **train_params)
    test_loader = DataLoader(test_dataset, **test_params)

    return train_loader, test_loader
