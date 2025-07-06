# scripts/train.py

import os
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from satellite_dataset import CLIPSatelliteDataset
import open_clip


def train(
    metadata_csv,
    image_root,
    model_name='ViT-B-32',
    pretrained='openai',
    subset_size=5000,
    batch_size=16,
    num_epochs=5,
    lr=5e-5,
    save_path="../models",
    device=None
):
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on device:", device)

    # Load model, transforms, tokenizer
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)

    # Dataset and Dataloader
    dataset = CLIPSatelliteDataset(
        metadata_csv=metadata_csv,
        image_root=image_root,
        transform=preprocess,
        tokenizer=tokenizer,
        max_samples=subset_size
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.train()

    os.makedirs(save_path, exist_ok=True)

    for epoch in range(num_epochs):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, texts in loop:
            images, texts = images.to(device), texts.to(device)

            # Forward pass
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            # Normalize
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # Similarity logits
            logits = image_features @ text_features.T

            # Symmetric InfoNCE loss
            labels = torch.arange(len(logits)).to(device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            loss = (loss_i2t + loss_t2i) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        # Save model
        save_file = os.path.join(save_path, f"clip_finetuned_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), save_file)
        print(f"Model saved to {save_file}")


if __name__ == "__main__":
    # Paths relative to your repo root
    metadata_csv = "../data/EuroSAT_RGB/metadata.csv"
    image_root = "../data/EuroSAT_RGB"

    train(
        metadata_csv=metadata_csv,
        image_root=image_root,
        subset_size=5000,    # Change to None to use full dataset
        num_epochs=5,
        batch_size=16,
        save_path="../models"
    )
