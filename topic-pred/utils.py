import re
import operator
import time
import os
import pdb
import wandb
import numpy as np
from nltk.tokenize import word_tokenize


import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchmetrics

from sklearn.metrics import average_precision_score as aps
from arguments import parse_arguments
torch.manual_seed(123)
torch.cuda.manual_seed(123)



DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Training on:", DEVICE)


def custom_train_test_split(data, test_size=0.2):
    """ Splits the input corpus in a train and a test set
    :param text: input corpus
    :param test_size: size of the test set, in fractions of the original corpus
    :return: train and test set
    """
    k = int(len(data) * (1 - test_size))
    return data[:k], data[k:]

def reverse_tag2poem(obj_dict):
    poem_tag_dict = {}
    for tag, poems in obj_dict.items():
        for poem in poems:
            if poem in poem_tag_dict:
                if tag in poem_tag_dict[poem]:
                    pass
                else:
                    poem_tag_dict[poem].append(tag)
            else:    
                poem_tag_dict[poem] = [tag]
    return poem_tag_dict

def pre_process(text):
    text = text.lower()
    text = text.replace("\r", "").strip('\n').strip()
    text = re.sub('\n+', '\n',text)
    text = text.replace("\n ", "")
    # text = text.replace("\n", " <eos> ")
    text = re.sub(' +', ' ',text)
    text = word_tokenize(text)
    return text

def pre_process_tags(tags):
    tags = tags.lower()
    tags = tags.split(",")
    tags = [x.strip() for x in tags]
    return tags

def unique_tags(tags, hl_tags=None):
    tag_freq = {}
    for tag in list(tags):
        items = tag
        for item in items:
            item = item.lower().strip()
            if hl_tags:
                if item in hl_tags:        
                    if item in tag_freq:
                        tag_freq[item] += 1
                    else:
                        if item !="":
                            tag_freq[item] = 1
            else:
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
        train = TensorDataset(torch.LongTensor(data['x_train']), 
                              torch.LongTensor(data['y_train']), 
                              torch.LongTensor(data['seq_len_train']))
        
        test =  TensorDataset(torch.LongTensor(data['x_test']), 
                              torch.LongTensor(data['y_test']), 
                              torch.LongTensor(data['seq_len_test']))
        
        valid = TensorDataset(torch.LongTensor(data['x_valid']), 
                              torch.LongTensor(data['y_valid']), 
                              torch.LongTensor(data['seq_len_valid']))
        
        # Define params
        
        # Initialize loaders
        loader_train = DataLoader(dataset=train, batch_size=params.batch_size, shuffle=True)
        loader_valid = DataLoader(dataset=valid, batch_size=params.batch_size, shuffle=False)
        loader_test = DataLoader(dataset=test, batch_size=params.batch_size, shuffle=False)
        
        # Define optimizer and loss function
        optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate)
        # optimizer = optim.SGD(model.parameters(), lr=params.learning_rate)
        # optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
        
        # criterion = nn.CrossEntropyLoss(reduction='sum')
        criterion = nn.CrossEntropyLoss(reduction='sum')
        # criterion = nn.BCEWithLogitsLoss()

        # Tracking best validation accuracy
        best_accuracy = 0
        
        print("\nStart training...")
        print(f"{'Epoch':^7} | {'Train Acc':^10} | {'F1 Train':^10} | {'Val Loss':^10} | {'Val Acc':^9} | {'Test Acc':^9} | {'F1 Test':^10} | {'Elapsed':^9}")
        print("-"*100)
    
        # Starts training phase
        train_loss_list = []
        val_loss_list = []
        test_acc_list = []
        fscore_train = []
        fscore_test = []
        torch_F1 = torchmetrics.F1(num_classes=params.output_dim, average="micro")
        if params.high_level_tags:
            wandb.init(project="[HLT] Poem topic classification")    
        else:
            wandb.init(project="Poem topic classification")
        
        # Wand Configuration update
        param_dict = {}
        for key in list(vars(params)["__annotations__"].keys()):
            param_dict[key] = params.__dict__[key]
        wandb.config.update(param_dict)
        
        for epoch in range(params.epochs):
            # =======================================
            #               Training
            # =======================================

            # Tracking time and loss
            t0_epoch = time.time()
            total_loss = 0
            train_accuracy = 0
            f1_train = 0

            # Put the model into training mode
            model.train()
        
            # Starts batch training
            for x_batch, y_batch, seq_len in loader_train:
                # Load batch to GPU
                x_batch = x_batch.long()
                y_batch = y_batch.long()
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                # Feed the model
                y_pred = model(x_batch)
                
                # Compute loss and accumulate the loss values
                # pdb.set_trace()
                loss = criterion(y_pred, y_batch)
                total_loss += loss.item()

                # Clean gradients
                optimizer.zero_grad()

                # Gradients calculation
                loss.backward()
            
                # Gradients update
                optimizer.step()
                
                # Accumulate training accuracy (for all batch in one epoch)
                corrects = (torch.max(y_pred, 1)[1].view(y_batch.size()).data == y_batch.data).sum()
                acc = 100.0 * corrects/loader_train.batch_size
                train_accuracy += acc
                
                # Compute F1 Score over a batch
                f1_train += torch_F1(y_pred.detach().cpu(), y_batch.detach().cpu())

            # Calculate the average loss over the entire training data for an epoch
            avg_train_loss = total_loss / len(loader_train)
            train_loss_list.append(avg_train_loss)
            
            # Compute accuracy and F1 averaged over all batches for an epoch
            train_accuracy = train_accuracy / len(loader_train)
            f1_train /= len(loader_train)
            fscore_train.append(f1_train)
            
            # =======================================
            #               Evaluation
            # =======================================
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            val_loss, val_accuracy, f1_val = evaluationn(model, loader_valid, params)
            _, test_accuracy, f1_test = evaluationn(model, loader_test, params)
            val_loss_list.append(val_loss)
            test_acc_list.append(test_accuracy)
            fscore_test.append(f1_test)
            
            # Track the best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                # Save the best model
                os.makedirs(params.model_dir, exist_ok=True)
                if params.save_model:
                    # Save the best model
                    PATH = params.model_dir + "/bi-dir_lstm_" + str(epoch+1) +"_batch_size_" + str(params.batch_size) + ".pth"
                    torch.save(model.state_dict(), PATH)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch + 1:^7} | {train_accuracy:^10.2f} | {f1_train:^10.4f} | {val_loss:^10.3f} | {val_accuracy:^9.2f} | {test_accuracy:^9.2f} | {f1_test:^10.4f} | {time_elapsed:^9.2f}")
            wandb.log({'train_acc': train_accuracy, 
                   'train_loss': avg_train_loss,
                   'val_acc': val_accuracy,
                   'val_loss': val_loss,
                   'test_acc': test_accuracy,
                   'f1_train': f1_train,
                   'f1_val': f1_val,
                   'f1_test': f1_test
                   })
            
            
        print("\n")
        print(f"Training complete! \nBest val accuracy: {best_accuracy:.2f} %.")
        os.makedirs(params.model_dir, exist_ok=True)
        if params.save_model:
            # Save the best model
            PATH = params.model_dir + "/cnn_" + str(epoch+1) +"_batch_size_" + str(params.batch_size) + ".pth"
            torch.save(model.state_dict(), PATH)
        
        print("Best model saved as: ", PATH)
        
        return train_loss_list, val_loss_list, test_acc_list, fscore_train, fscore_test, PATH
    
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

def evaluationn(model, loader_eval, params):
    # Set the model in evaluation mode
    model.eval()
    corrects, avg_loss, f1 = 0, 0, 0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    # Start evaluation phase   
    torch_F1 = torchmetrics.F1(num_classes=params.output_dim, average="micro")
    with torch.no_grad():
        for x_batch, y_batch, seq_len in loader_eval:
            x_batch = x_batch.long()
            y_batch = y_batch.long()
            
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            y_pred = model(x_batch)
            
            loss = criterion(y_pred, y_batch)
            avg_loss += loss.item()
            corrects += (torch.max(y_pred, 1) [1].view(y_batch.size()).data == y_batch.data).sum()
            f1 += torch_F1(y_pred.detach().cpu(), y_batch.detach().cpu())
            
    size = len(loader_eval.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    f1 /= len(loader_eval)
    
    return avg_loss, accuracy, f1