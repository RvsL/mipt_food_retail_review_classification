import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from transformers import logging as transformer_logging
transformer_logging.set_verbosity_error()

from bert_dataset import CustomDataset

class BertClassifier:

    def __init__(self, model_path, tokenizer_path, n_classes=2, epochs=1, model_save_path='/content/bert.pt', distil = False):

        if distil:
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
            self.out_features = self.model.pre_classifier.out_features
        else:
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            self.out_features = self.model.bert.encoder.layer[1].output.dense.out_features

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
        self.model_save_path=model_save_path
        self.max_len = 512
        self.epochs = epochs
        self.model.classifier = torch.nn.Linear(self.out_features, n_classes)
        self.model.to(self.device)
    
    def preparation(self, X_train, y_train, X_valid, y_valid, bsize=2):
        # create datasets
        self.train_set = CustomDataset(X_train, y_train, self.tokenizer)
        self.valid_set = CustomDataset(X_valid, y_valid, self.tokenizer)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=bsize, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=bsize, shuffle=True)

        # helpers initialization
#        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=0,
                num_training_steps=len(self.train_loader) * self.epochs
            )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
            
    def fit(self, progress_bar_train):
        self.model = self.model.train()
        losses = []
        correct_predictions = 0

        for data in self.train_loader:
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
                )

            preds = torch.argmax(outputs.logits, dim=1)
            loss = self.loss_fn(outputs.logits, targets)

            correct_predictions += torch.sum(preds == targets)

            losses.append(loss.item())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            progress_bar_train.update(1)

        train_acc = correct_predictions.float() / len(self.train_set)
        train_loss = np.mean(losses)
        return train_acc, train_loss
    
    def eval(self):
        self.model = self.model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for data in self.valid_loader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                targets = data["targets"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                    )

                preds = torch.argmax(outputs.logits, dim=1)
                loss = self.loss_fn(outputs.logits, targets)
                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())
        
        val_acc = correct_predictions.float() / len(self.valid_set)
        val_loss = np.mean(losses)
        return val_acc, val_loss
    
    def train(self):
        best_accuracy = 0
        losses = []
        
        num_training_steps = self.epochs * len(self.train_loader)
        progress_bar_train = tqdm(range(num_training_steps))

        for epoch in range(self.epochs):
#            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_acc, train_loss = self.fit(progress_bar_train)
#            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = self.eval()
#            print(f'Val loss {val_loss} accuracy {val_acc}')
#            print('-' * 10)
            losses.append([epoch + 1, train_loss, val_loss, val_acc])

            if val_acc > best_accuracy:
                torch.save(self.model, self.model_save_path)
                best_accuracy = val_acc

        losses = pd.DataFrame(losses, columns=['epoch', 'loss_train', 'loss_val', 'val_acc'])
        self.model = torch.load(self.model_save_path)

        plt.figure(figsize=(15,5))
        
        plt.subplot(121)
        plt.plot(losses['epoch'], losses['loss_train'], '-o')
        plt.plot(losses['epoch'], losses['loss_val'], '-o')
        
        plt.xlabel('epoch')
        plt.ylabel('loss')
        
        plt.legend(['train loss', 'val loss'])
        plt.title('Loss - epoch dependency')
        
        
        plt.subplot(122)
        plt.plot(losses['epoch'], losses['val_acc'], '-o')
        
        plt.xlabel('epoch')
        plt.ylabel('val accuracy')
        plt.title('val accuracy - epoch dependency')
            
        plt.show()

    def predict(self, text):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        out = {
              'text': text,
              'input_ids': encoding['input_ids'].flatten(),
              'attention_mask': encoding['attention_mask'].flatten()
          }
        
        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)
        
        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )
        
        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

        return prediction
