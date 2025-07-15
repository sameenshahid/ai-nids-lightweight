import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import joblib
import os

# === Step 1: Load Data ===
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# === Step 2: Load Tokenizer and Model ===
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
model_path = r"D:\Downloads\faizan\faizan\results\full_model"
model = BertForSequenceClassification.from_pretrained(model_path, output_hidden_states=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Step 3: Function to Get CLS Embeddings ===
# Use this to get CLS embeddings (same as in training)
def get_cls_embeddings(texts, batch_size=32):
    cls_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encodings = tokenizer(batch_texts.tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt")
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_batch = outputs.hidden_states[-1][:, 0, :]  # CLS token from last hidden layer
            cls_embeddings.append(cls_batch.cpu().numpy())
    return np.vstack(cls_embeddings)

# === Step 4: File name for saving/loading model ===
model_filename = "tinybert_lgbm_pipeline.pkl"

if os.path.exists(model_filename):
    # === Load Saved Model and Predict ===
    print(f"[INFO] Loading model from {model_filename}")
    best_model = joblib.load(model_filename)

    # Get test embeddings
    print("[INFO] Extracting CLS embeddings for test set...")
    X_test = get_cls_embeddings(test_df['text'])
    y_test = test_df['label'].values

    preds = best_model.predict(X_test)
    print("\n[INFO] Classification Report on Test Set:")
    print(classification_report(y_test, preds))

else:
    # === Train New Model ===
    print("[INFO] Extracting CLS embeddings for train and test sets...")
    X_train = get_cls_embeddings(train_df['text'])
    y_train = train_df['label'].values
    X_test = get_cls_embeddings(test_df['text'])
    y_test = test_df['label'].values

    # Pipeline: PCA + LightGBM
    pipeline = Pipeline([
        ('pca', PCA(n_components=50)),
        ('lgbm', LGBMClassifier(random_state=42))
    ])

    param_grid = {
        'pca__n_components': [50, 100],
        'lgbm__n_estimators': [100, 200],
        'lgbm__learning_rate': [0.1, 0.05],
        'lgbm__max_depth': [5, 10]
    }

    print("[INFO] Starting GridSearchCV...")
    grid = GridSearchCV(pipeline, param_grid, scoring='f1_weighted', cv=3, verbose=2, n_jobs=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"\n[INFO] Best Params:\n{grid.best_params_}")

    X_test = get_cls_embeddings(test_df['text'])  # ✅ Correct embeddings
    y_test = test_df['label'].values
    

    preds = best_model.predict(X_test)
    print("\n[INFO] Classification Report on Test Set:")
    print(classification_report(y_test, preds))

    # Save model
    joblib.dump(best_model, model_filename)
    print(f"[INFO] Trained model saved as {model_filename}")
