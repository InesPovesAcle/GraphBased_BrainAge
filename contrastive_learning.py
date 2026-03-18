#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 18:02:50 2026

@author: ines
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SHAP Contrastive Learning
Train embedding model using SHAP vectors and triplets
"""

################# IMPORTS ################

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

################# PATHS ################

WORK = os.environ["WORK"]

# Input directory
INPUT_DIR = os.path.join(WORK, "ines/results/Shap_cvs_global_features")

# Output directory (NEW)
OUTPUT_DIR = os.path.join(WORK, "ines/results/contrastive_learning_addecode_shap")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SHAP_ZSCORED_PATH = os.path.join(INPUT_DIR, "shap_global_features_zscored.csv")
TRIPLETS_PATH = os.path.join(INPUT_DIR, "shap_triplets_topk.csv")

MODEL_OUT = os.path.join(OUTPUT_DIR, "shap_projection_head_trained.pt")
EMBEDDINGS_OUT = os.path.join(OUTPUT_DIR, "shap_embeddings.csv")

print("Input directory:", INPUT_DIR)
print("Output directory:", OUTPUT_DIR)

################# REPRODUCIBILITY ################

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

################# LOAD SHAP VECTORS ################

df_shap_z = pd.read_csv(SHAP_ZSCORED_PATH)

df_shap_z["Subject_ID"] = df_shap_z["Subject_ID"].astype(str).str.zfill(5)

subject_ids = df_shap_z["Subject_ID"].values
shap_vectors = df_shap_z.drop(columns=["Subject_ID"]).values.astype(np.float32)

id_to_vector = {subj: vec for subj, vec in zip(subject_ids, shap_vectors)}

print("SHAP input dimension:", shap_vectors.shape[1])
print("Example subject:", subject_ids[0])

################# LOAD TRIPLETS ################

triplet_df = pd.read_csv(TRIPLETS_PATH)

triplet_df["anchor"] = triplet_df["anchor"].astype(str).str.zfill(5)
triplet_df["positive"] = triplet_df["positive"].astype(str).str.zfill(5)
triplet_df["negative"] = triplet_df["negative"].astype(str).str.zfill(5)

triplets = list(zip(
    triplet_df["anchor"],
    triplet_df["positive"],
    triplet_df["negative"]
))

print("Triplets loaded:", len(triplets))

################# DATASET ################

class TripletDataset(Dataset):

    def __init__(self, triplets, id_to_vector):
        self.triplets = triplets
        self.id_to_vector = id_to_vector

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):

        anchor_id, pos_id, neg_id = self.triplets[idx]

        anchor = torch.tensor(self.id_to_vector[anchor_id], dtype=torch.float32)
        positive = torch.tensor(self.id_to_vector[pos_id], dtype=torch.float32)
        negative = torch.tensor(self.id_to_vector[neg_id], dtype=torch.float32)

        return anchor, positive, negative


dataset = TripletDataset(triplets, id_to_vector)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

print("Dataset size:", len(dataset))

################# MODEL ################

input_dim = shap_vectors.shape[1]

class ShapProjectionHead(nn.Module):

    def __init__(self, input_dim=input_dim, hidden_dim=64, output_dim=32):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.normalize(x, p=2, dim=1)


class TripletNTXentLoss(nn.Module):

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negative):

        sim_ap = torch.sum(anchor * positive, dim=1) / self.temperature
        sim_an = torch.sum(anchor * negative, dim=1) / self.temperature

        logits = torch.stack([sim_ap, sim_an], dim=1)

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        return loss

################# TRAIN ################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = ShapProjectionHead(input_dim=input_dim).to(device)

criterion = TripletNTXentLoss(temperature=0.5)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5
)

epochs = 100

model.train()

for epoch in range(epochs):

    total_loss = 0

    for anchor, positive, negative in dataloader:

        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()

        anchor_embed = model(anchor)
        positive_embed = model(positive)
        negative_embed = model(negative)

        loss = criterion(anchor_embed, positive_embed, negative_embed)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

################# SAVE MODEL ################

torch.save(model.state_dict(), MODEL_OUT)

print("Model saved:", MODEL_OUT)

################# SAVE EMBEDDINGS ################

model.eval()

all_embeddings = {}

with torch.no_grad():

    for subj_id, shap_vec in id_to_vector.items():

        tensor_input = torch.tensor(shap_vec, dtype=torch.float32).unsqueeze(0).to(device)

        embedding = model(tensor_input).squeeze(0).cpu().numpy()

        all_embeddings[subj_id] = embedding

embedding_df = pd.DataFrame.from_dict(all_embeddings, orient="index")

embedding_df.index.name = "Subject_ID"

embedding_df.columns = [f"embed_{i}" for i in range(embedding_df.shape[1])]

embedding_df = embedding_df.reset_index()

embedding_df.to_csv(EMBEDDINGS_OUT, index=False)

print("Embeddings saved:", EMBEDDINGS_OUT)

print("\nDONE.")