from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, AutoModel, AutoConfig
from torch.utils.data import DataLoader, Dataset
import torch
import tqdm
import os
from torch.cuda.amp import GradScaler, autocast
from methods.utils import *
import torch.nn.functional as F


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

def fine_tune_radar(model, tokenizer, data, batch_size, DEVICE, epochs=3, ckpt_dir='./ckpt', three_classes=False):
    train_texts = data['train']['text']
    # train_labels = data['train']['label']
    new_labels = []
    for i in data['train']['label']:
        if i == 0:
            new_labels.append(1)
        elif i == 1:
            new_labels.append(0)
        elif i == 2:
            new_labels.append(2)
    train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=512)
    train_dataset = CustomDataset(train_encodings, new_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-6)
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
    model.save_pretrained(os.path.join(ckpt_dir, 'radar-finetuned'))
    print(f"Saved finetuned model to {os.path.join(ckpt_dir, 'radar')}")

def evaluate_model(model, tokenizer, data, DEVICE, no_auc=False):
    sentences = data['test']['text']
    labels = data['test']['label']

    probs = []
    preds = []
    errors = 0  # to keep track of sentences that caused errors

    with torch.no_grad():
        for i in tqdm.tqdm(range(len(sentences)), desc="Evaluating"):
            try:
                inputs = tokenizer(sentences[i], return_tensors="pt").to(DEVICE)
                logits = model(**inputs).logits.to('cpu')
                softmax_probs = F.softmax(logits, dim=-1)

                class_0_probs = softmax_probs[0].tolist()[0]
                probs.append(class_0_probs)

                predicted_class_id = int(logits.argmax())
                if model.config.id2label[predicted_class_id] == "LABEL_0":
                    preds.append(1)
                elif model.config.id2label[predicted_class_id] == "LABEL_1":
                    preds.append(0)
                elif model.config.id2label[predicted_class_id] == "LABEL_2":
                    preds.append(2)
            
            except RuntimeError as e:
                print(f"Error processing sentence index {i}: {str(e)}")
                errors += 1
                probs.append(0.5)
                preds.append(1)
                continue  # Skip this sentence and move to the next

    if errors > 0:
        print(f"Encountered {errors} errors during evaluation.")

    test_res = cal_metrics(labels, preds, probs, no_auc=no_auc)
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

def run_radar(data, DEVICE, finetune: bool=False, no_auc: bool=False, ckpt_dir = './ckpt', test_only: bool=False, three_classes: bool=False):
    if three_classes:
        config = AutoConfig.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B", num_labels=3)
    else: 
        config = AutoConfig.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
    tokenizer = AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
    model = AutoModelForSequenceClassification.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B", config=config, ignore_mismatched_sizes=True).to(DEVICE)
    if not three_classes:
        dict_before = evaluate_model(model, tokenizer, data, DEVICE, no_auc=no_auc)
    
    if finetune and not test_only:
        fine_tune_radar(model, tokenizer, data, batch_size=12, DEVICE=DEVICE, epochs=3, ckpt_dir = ckpt_dir, three_classes=three_classes)
    elif finetune and test_only:
        model = AutoModelForSequenceClassification.from_pretrained(os.path.join(ckpt_dir, 'radar-finetuned')).to(DEVICE)
    
    dict_after = evaluate_model(model, tokenizer, data, DEVICE, no_auc=no_auc)
    
    return {"Radar_retrained" if finetune else "Radar": dict_after}