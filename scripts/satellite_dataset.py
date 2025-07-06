import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import random
import torch


class CLIPSatelliteDataset(Dataset):
    def __init__(self, metadata_csv, image_root, transform, tokenizer, max_samples=None, seed=42):
        self.df = pd.read_csv(metadata_csv)
        self.image_root = image_root
        self.transform = transform
        self.tokenizer = tokenizer

        if max_samples:
            self.df = self.df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_root, row['filename'])
        text = row['text']

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        tokenized = self.tokenizer([text])[0]

        return image, tokenized
