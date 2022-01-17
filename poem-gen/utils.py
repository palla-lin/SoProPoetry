#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:  Sun Jan 16 11:30:35 AM CET 2022


import time
import os
import pdb
import wandb
import numpy as np
import time
import datetime
import glob
import random

import torch
import torch.optim as optim
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup

RANDOM_SEED = random.randint(100, 999)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Training on:", DEVICE)


# helper function for logging time
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


class Run(object):
    '''Training, evaluation and metrics calculation'''

    @staticmethod
    def train(tokenizer, train_dataloader, val_dataloader, params):

        configuration = GPT2Config(vocab_size=len(
            tokenizer), n_positions=params.MAX_LEN).from_pretrained('gpt2', output_hidden_states=True)

        model = GPT2LMHeadModel.from_pretrained('gpt2', config=configuration)
        model.resize_token_embeddings(len(tokenizer))

        model.cuda()
        optimizer = AdamW(model.parameters(),
                        lr=params.learning_rate, eps=params.eps)
        total_steps = len(train_dataloader) * params.EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=params.warmup_steps,
                                                    num_training_steps=total_steps)
        start_time = time.time()
        model = model.to(DEVICE)

        for epoch_i in range(0, params.EPOCHS):
            print(f'Epoch {epoch_i + 1} of {params.EPOCHS}')
            t0 = time.time()
            total_train_loss = 0
            model.train()

            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(DEVICE)
                b_labels = batch[0].to(DEVICE)
                b_masks = batch[1].to(DEVICE)

                model.zero_grad()
                outputs = model(b_input_ids,
                                labels=b_labels,
                                attention_mask=b_masks,
                                token_type_ids=None)

                loss = outputs[0]

                batch_loss = loss.item()
                total_train_loss += batch_loss

                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            training_time = format_time(time.time() - t0)
            print(f'Average Training Loss: {avg_train_loss}. Epoch Training Time: {training_time}')

            t0 = time.time()
            model.eval()
            total_eval_loss = 0
            nb_eval_steps = 0

            for batch in val_dataloader:
                b_input_ids = batch[0].to(DEVICE)
                b_labels = batch[0].to(DEVICE)
                b_masks = batch[1].to(DEVICE)
                with torch.no_grad():
                    outputs = model(b_input_ids,
                                    attention_mask=b_masks,
                                    labels=b_labels)
                    loss = outputs[0]

                batch_loss = loss.item()
                total_eval_loss += batch_loss

            avg_val_loss = total_eval_loss / len(val_dataloader)
            print(f'Average Validation Loss: {avg_val_loss}')

        print(f'Total Training Time: {format_time(time.time()-start_time)}')
        torch.save(model.state_dict(), params.out_dir + '/model.pth')

class ConditionalGenerate():
    
    @staticmethod
    def generate(tokenizer, params):

        # Load trained model
        configuration = GPT2Config(vocab_size=len(
            tokenizer), n_positions=params.MAX_LEN).from_pretrained('gpt2', output_hidden_states=True)
        
        model = GPT2LMHeadModel.from_pretrained('gpt2', config=configuration)
        model.resize_token_embeddings(len(tokenizer))

        model_path = glob.glob(params.out_dir + "/*.pth")[0]
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.cuda()        
        model.eval()

        
        ip_topic = ''
        while ip_topic != 'exit':
            ip_topic = str(input("Enter poem topic: "))
            prompt = "<BOS> " + ip_topic
            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
            generated = generated.to(DEVICE)
            
            
            sample_outputs = model.generate(
                                            generated, 
                                            do_sample=True,   
                                            top_k=50, 
                                            max_length=500,
                                            top_p=0.95, 
                                            num_return_sequences=3
                                            )
            print("*"*20 + str(ip_topic.upper()) + " poems" + "*"*20)
            for i, sample_output in enumerate(sample_outputs):
                print("{}: {}\n\n".format(i+1, tokenizer.decode(sample_output, skip_special_tokens=True)))
                print("-"*40)





