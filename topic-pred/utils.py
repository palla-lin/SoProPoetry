import re
import operator
import time
import os
import pdb
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchmetrics

from sklearn.metrics import average_precision_score as aps
from arguments import parse_arguments
args = parse_arguments()

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Training on:", DEVICE)

def pre_process(text):
    text = text.lower()
    text = text.replace("\r", "").strip('\n').strip()
    text = re.sub('\n+', '\n',text)
    text = text.replace("\n ", "")
    text = text.replace("\n", " <eos> ")
    text = re.sub(' +', ' ',text)
    return text

def pre_process_tags(tags):
    tags = tags.lower()
    tags = tags.split(",")
    tags = [x.strip() for x in tags]
    return tags

def unique_tags(tags):
    tag_freq = {}
    for tag in tags:
        items = tag.split(",")
        for item in items:
            item = item.lower().strip()
            if item in tag_freq:
                tag_freq[item] += 1
            else:
                if item !="":
                    tag_freq[item] = 1
    
    return dict(sorted(tag_freq.items(),key=operator.itemgetter(1),reverse=True))

def tag_mapper(tags, tag2int):
    int_tags = []
    for i in tags:
        int_tags.append(tag2int[i])
    return int_tags

def load_glove_model(File):
    glove_model = {}
    with open(File,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model


class DatasetMaper(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Run:
    '''Training, evaluation and metrics calculation'''

    @staticmethod
    def train(model, data, params):
        train = DatasetMaper(data['x_train'], data['y_train'])
        test =  DatasetMaper(data['x_test'], data['y_test'])
        valid = DatasetMaper(data['x_valid'], data['y_valid'])
        
        # Define params
        
        # Initialize loaders
        loader_train = DataLoader(dataset=train, batch_size=params.batch_size, shuffle=True)
        loader_valid = DataLoader(dataset=valid, batch_size=params.batch_size, shuffle=False)
        loader_test = DataLoader(dataset=test, batch_size=params.batch_size, shuffle=False)
        
        # Define optimizer and loss function
        optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate)
        # optimizer = optim.SGD(model.parameters(), lr=params.learning_rate)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # criterion = nn.CrossEntropyLoss(reduction='sum')
        criterion = nn.BCEWithLogitsLoss()

        # Tracking best validation accuracy
        best_accuracy = 0
        
        print("\nStart training...")
        print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Elapsed':^9}")
        # print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^9} | {'Test Acc':^9} | {'Elapsed':^9}")
        print("-"*80)
    
        # Starts training phase
        train_loss_list = []
        val_loss_list = []
        test_acc_list = []
        
        for epoch in range(params.epochs):
            # =======================================
            #               Training
            # =======================================

            # Tracking time and loss
            t0_epoch = time.time()
            total_loss = 0
            train_accuracy = 0

            # Put the model into training mode
            model.train()
        
            # Starts batch training
            for x_batch, y_batch in loader_train:
                
                # Load batch to GPU
                x_batch = x_batch.long()
                y_batch = y_batch.to(torch.float32)
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                # Feed the model
                y_pred = model(x_batch)

                # Compute loss and accumulate the loss values
                loss = criterion(y_pred, y_batch)
                total_loss += loss.item()

                # Clean gradients
                optimizer.zero_grad()

                # Gradients calculation
                loss.backward()

                # Gradients update
                optimizer.step()
                
                # Compute multi-label metrics on one batch
                # multi_matric = calculate_metrics(y_batch, y_pred)

            # Calculate the average loss over the entire training data for a batch
            avg_train_loss = total_loss / len(loader_train)
            train_loss_list.append(avg_train_loss)
            
            # =======================================
            #               Evaluation
            # =======================================
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            
            # Validation metrics
            val_loss = evaluation(model, loader_valid)
            val_loss_list.append(val_loss)
            
            # Track the best accuracy
            # if val_accuracy > best_accuracy:
            #     best_accuracy = val_accuracy
            #     # Save the best model
            #     os.makedirs(params.model_dir, exist_ok=True)
            #     if params.save_model:
            #         # Save the best model
            #         PATH = params.model_dir + "/bi-dir_lstm_" + str(epoch+1) +"_batch_size_" + str(params.batch_size) + ".pth"
            #         torch.save(model.state_dict(), PATH)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch + 1:^7} | {avg_train_loss:^12.6f}  | {val_loss:^10.6f} | {time_elapsed:^9.2f}")
            # print(f"{epoch + 1:^7} | {avg_train_loss:^12.6f} | {train_accuracy:^10.2f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {test_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            
            
        print("\n")
        print(f"Training complete! \nBest accuracy: {best_accuracy:.2f} %.")
        os.makedirs(params.model_dir, exist_ok=True)
        if params.save_model:
            # Save the best model
            PATH = params.model_dir + "/bi-dir_lstm_" + str(epoch+1) +"_batch_size_" + str(params.batch_size) + ".pth"
            torch.save(model.state_dict(), PATH)
        
        return train_loss_list, val_loss_list
    
def batch_accuracy(y_pred, y_batch):
    y_pred = torch.round(torch.sigmoid(y_pred)).cpu().detach()
    y_batch = y_batch.cpu()
    acc = (y_pred== y_batch).sum().item()/(y_pred.size()[0]* y_pred.size()[1])
    return acc

        
    
def calculate_metrics(y_batch, y_pred):
    # f1 = torchmetrics.F1(threshold=0.5, num_classes=None, average='micro')
    # pre = torchmetrics.Precision(threshold=0.5, num_classes=None, average='micro')
    # rec = torchmetrics.Recall(threshold=0.5, num_classes=None, average='micro')
    
    y_pred = torch.round(torch.sigmoid(y_pred)).cpu().detach()
    y_batch = y_batch.cpu()
    pdb.set_trace()
    waps = aps(y_batch, y_pred, average='weighted', sample_weight='samples')
    
    return {
        "waps":  waps
    }

def evaluation(model, loader_eval):

    # Set the model in evaluation mode
    model.eval()
    avg_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    # Start evaluation phase   
    with torch.no_grad():
        for x_batch, y_batch in loader_eval:
            x_batch = x_batch.long()
            y_batch = y_batch.to(torch.float32)
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            avg_loss += loss.item()
            
    size = len(loader_eval.dataset)
    avg_loss /= size
    return avg_loss