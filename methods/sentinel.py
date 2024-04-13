from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW
import os
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

def evaluate_sentinel(model, tokenizer, data, DEVICE, ckpt_dir='./ckpt', no_auc=False):
    sentences = data['test']['text']
    labels = data['test']['label']
    probs = []
    preds = []
    model.eval()
    model.to(DEVICE)
    for i in tqdm.tqdm(range(len(sentences)), desc="Sentinel evaluating"):
        input_ids = tokenizer.encode(sentences[i], return_tensors='pt').to(DEVICE)
        output = model.generate(input_ids)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

        logits = model(input_ids, decoder_input_ids=output, return_dict=True).logits[0][0]
        positive_idx = tokenizer.convert_tokens_to_ids('positive')
        negative_idx = tokenizer.convert_tokens_to_ids('negative')

        new_logits = torch.full_like(logits, float('-inf'))
        new_logits[positive_idx] = logits[positive_idx]
        new_logits[negative_idx] = logits[negative_idx]

        softmax_probs = F.softmax(new_logits, dim=-1)
        positive_prob = softmax_probs[positive_idx].item()
        negative_prob = softmax_probs[negative_idx].item()
        
        probs.append(positive_prob)
        preds.append(1 if positive_prob > negative_prob else 0)
        
    test_res = cal_metrics(labels, preds, probs, no_auc=no_auc)
    acc_test, precision_test, recall_test, f1_test, auc_test = test_res

    print(f"acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")
    results_dict = {
        "name": "GPT-sentinel", 
        'acc_test': acc_test,
        'precision_test': precision_test,
        'recall_test': recall_test,
        'f1_test': f1_test,
        'auc_test': auc_test,
    }
    print(results_dict)
    return results_dict

def run_sentinel(data, DEVICE, finetune: bool=False, no_auc: bool=False, ckpt_dir = './ckpt', test_only: bool=False):
    model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    state_dict = torch.load('t5.small.0422.pt')['model']
    adjusted_state_dict = {k.replace('t5_model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(adjusted_state_dict, strict=True)
    
    dict_before = evaluate_sentinel(model, tokenizer, data, DEVICE, ckpt_dir=ckpt_dir, no_auc=no_auc)
    
    if finetune and not test_only:
        fine_tune_model(model, tokenizer, data, batch_size=16, DEVICE=DEVICE, epochs=3, ckpt_dir = ckpt_dir)
    elif finetune and test_only:
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'sentinel.pth')))
        
    dict_after = evaluate_sentinel(model, tokenizer, data, DEVICE, ckpt_dir=ckpt_dir, no_auc=no_auc)
    return {"after": dict_after}


def fine_tune_model(model, tokenizer, data, batch_size, DEVICE, epochs=3, ckpt_dir='./ckpt'):
    train_text = ['classify: ' + text for text in data['train']['text']]
    train_label = ['positive' if label == 1 else 'negative' for label in data['train']['label']]

    train_encodings = tokenizer(train_text, truncation=True, padding='longest', pad_to_multiple_of=512)
    train_labels = tokenizer(train_label, padding='max_length', max_length=512, truncation=True).input_ids

    train_labels = [[-100] * (512 - 1) + [label[0]] for label in train_labels]
    
    train_dataset = CustomDataset(train_encodings, train_labels)

    model.train()
    model.to(DEVICE)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        model.train()
        for batch in tqdm.tqdm(train_loader, desc=f"Fine-tuning: {epoch} epoch"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    model.eval()

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'sentinel.pth'))
    print(f"Save trained model to: {os.path.join(ckpt_dir, 'sentinel.pth')}")