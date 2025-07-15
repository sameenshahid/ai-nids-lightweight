import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Load dataset
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv").sample(n=5000, random_state=42)

print("Unique labels in train:", sorted(train_df["label"].unique()))
print("Unique labels in test:", sorted(test_df["label"].unique()))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

# Dataset class
class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row["text"]
        label = row["label"]
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs["labels"] = torch.tensor(label, dtype=torch.long)
        return inputs

# DataLoaders
train_dataset = NewsDataset(train_df, tokenizer)
test_dataset = NewsDataset(test_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Class weights setup
class_labels = train_df["label"].values
num_labels = train_df["label"].max() + 1
present_classes = np.unique(class_labels)
weights_for_present = compute_class_weight(class_weight='balanced', classes=present_classes, y=class_labels)
class_weights = torch.ones(num_labels, dtype=torch.float)
for idx, cls in enumerate(present_classes):
    class_weights[cls] = weights_for_present[idx]

# Model and training setup
model = AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=num_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_fn = CrossEntropyLoss(weight=class_weights.to(device))
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
print("\nStarting training...")
model.train()
for epoch in range(3):
    print(f"\nEpoch {epoch+1}")
    total_loss = 0
    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

# Save model
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/tinybert_model.pth")
print("✅ Model state_dict saved to results/tinybert_model.pth")

# Save full model and tokenizer
model.save_pretrained("results/full_model")
tokenizer.save_pretrained("results/full_model")
print("✅ Full model and tokenizer saved to results/full_model/")


# Evaluation
print("\nStarting evaluation...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification Report
report = classification_report(all_labels, all_preds, digits=4)
print("\nClassification Report:\n", report)
with open("results/classification_report.txt", "w") as f:
    f.write(report)
print("✅ Classification report saved to results/classification_report.txt")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()
print("✅ Confusion matrix saved to results/confusion_matrix.png")
