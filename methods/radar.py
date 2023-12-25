from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, AutoModel
from torch.utils.data import DataLoader, Dataset
import torch
import tqdm
import os
from torch.cuda.amp import GradScaler, autocast
from utils import *

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def fine_tune_radar(model, tokenizer, data, batch_size, DEVICE, epochs=3, ckpt_dir='./ckpt'):
    train_texts = data['train']['text']
    # train_labels = data['train']['label']
    train_labels = [1 if label == 0 else 0 for label in data['train']['label']]
    train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=512)
    train_dataset = CustomDataset(train_encodings, train_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler() 
    model.train()
    model.to(DEVICE)
    for epoch in range(epochs):
        for batch in tqdm.tqdm(train_loader, desc=f"Fine-tuning: {epoch} epoch"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    model.save_pretrained(os.path.join(ckpt_dir, 'radar'))
    print(f"Saved finetuned model to {os.path.join(ckpt_dir, 'radar')}")

def evaluate_model(model, tokenizer, data, DEVICE, no_auc=False):
    sentences = data['test']['text']
    labels = data['test']['label']

    probs = []
    preds = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(sentences)), desc="Evaluating"):
            inputs = tokenizer(sentences[i], return_tensors="pt").to(DEVICE)
            logits = model(**inputs).logits.to('cpu')
            probs.append(logits.argmax().tolist()[0])
            predicted_class_id = int(logits.argmax())
            if model.config.id2label[predicted_class_id] == "LABEL_0":
                preds.append(1)
            elif model.config.id2label[predicted_class_id] == "LABEL_1":
                preds.append(0)
    
    train_res, test_res = cal_metrics(labels, preds, probs, no_auc=no_auc)
    acc_test, precision_test, recall_test, f1_test, auc_test = test_res
    results_dict = {
        "name": "Radar", 
        'acc_test': acc_test,
        'precision_test': precision_test,
        'recall_test': recall_test,
        'f1_test': f1_test,
        'auc_test': auc_test,
    }
    print(results_dict)
    return results_dict

def run_radar(data, DEVICE, finetune: bool=False, no_auc: bool=False, ckpt_dir = './ckpt', test_only: bool=False):
    tokenizer = AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
    model = AutoModelForSequenceClassification.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B").to(DEVICE)
    
    dict_before = evaluate_model(model, tokenizer, data, DEVICE, no_auc=no_auc)
    
    if finetune and not test_only:
        fine_tune_radar(model, tokenizer, data, batch_size=12, DEVICE=DEVICE, epochs=3, ckpt_dir = ckpt_dir)
    elif finetune and test_only:
        model = AutoModel.from_pretrained(os.path.join(ckpt_dir, 'radar'))
        
    dict_after = evaluate_model(model, tokenizer, data, DEVICE, no_auc=no_auc)
    
    return {"Radar_retrained" if finetune else "Radar": dict_after}