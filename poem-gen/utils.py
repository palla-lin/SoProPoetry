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
import json
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup

from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining, \
                         AdamW, get_linear_schedule_with_warmup, \
                         TrainingArguments, BeamScorer, Trainer

from dataset_loader import NeuralPoetDataset
import pickle

RANDOM_SEED = random.randint(100, 999)
torch.cuda.manual_seed_all(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
MODEL = 'gpt2'

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Training on:", DEVICE)

SPECIAL_TOKENS  = {
            'bos_token': '<|BOS|>',  # beginning of sents
            'eos_token': '<|EOS|>',  # end of sents
            'pad_token': '<|PAD|>',  # padding toks
            'sep_token': '<|SEP|>',  # separator
            'unk_token': '<|UNK|>'   # unknown token
        }


# helper function for logging time
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


class Run(object):
    '''Training, evaluation and metrics calculation'''

    @staticmethod
    def train(tokenizer, train_dataloader, val_dataloader, params):

        # configuration = GPT2Config(vocab_size=len(
        #     tokenizer), n_positions=params.MAX_LEN).from_pretrained('gpt2', output_hidden_states=True)
        
        configuration = AutoConfig.from_pretrained(MODEL,
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)
        
        # model = GPT2LMHeadModel.from_pretrained('gpt2', config=configuration)
        model = AutoModelForPreTraining.from_pretrained(MODEL, config=configuration)        
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
        configuration = AutoConfig.from_pretrained(MODEL,
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    sep_token_id=tokenizer.sep_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    output_hidden_states=False)
        
        model = AutoModelForPreTraining.from_pretrained(MODEL, config=configuration)
        model.resize_token_embeddings(len(tokenizer))

        model_path = glob.glob(params.out_dir + "/*.pth")[0]
        print("Using model: ", model_path)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.cuda()        
        model.eval()

        
        ip_topic = ''
        while ip_topic != 'exit':
            ip_topic = str(input("Enter poem topic: "))
            keywords = str(input("Enter list of keywords (separated by commas (,)): "))
            keywords = keywords.split(",")
            random.shuffle(keywords)
            kw = ','.join(keywords)
            
            condition = SPECIAL_TOKENS['bos_token'] + ip_topic + \
            SPECIAL_TOKENS['sep_token'] + kw + SPECIAL_TOKENS['sep_token']
        
            print(condition)
            
            generated = torch.tensor(tokenizer.encode(condition)).unsqueeze(0)
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
                text = tokenizer.decode(sample_output, skip_special_tokens=True)
                a = len(ip_topic) + len(','.join(keywords)) 
                print("{}: {}\n\n".format(i+1,  text[a:]))
                print("-"*40)


class ConditionalGenerateMultiPoems():
    
    @staticmethod
    def generate(data, tokenizer, params):

        # Load trained model
        configuration = AutoConfig.from_pretrained(MODEL,
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    sep_token_id=tokenizer.sep_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    output_hidden_states=False)
        
        model = AutoModelForPreTraining.from_pretrained(MODEL, config=configuration)
        model.resize_token_embeddings(len(tokenizer))

        model_path = glob.glob(params.out_dir + "/*.pth")[0]
        print("Using model: ", model_path)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.cuda()        
        model.eval()

        
        # We will randomly sample 3 keywords from each topic from the training set
        # and generate poems conditioned upon them
        tag_keywords = {}
        for idx, line in data.items():
            tag = line["tags"]
            kws = line["keywords"]
            if tag in tag_keywords:
                tag_keywords[tag].extend(kws)
            else:
                tag_keywords[tag] = kws
                
        for tag, words in tag_keywords.items():
            tag_keywords[tag] = list(set(words))
        
        flag = 0
        plo = 0
        to_write = {}
        for tag, kws in tag_keywords.items():
            flag += 1
            print("Generating poems for topic no. (144):", flag, end="\r")
            # plo += 1
            # if plo > 4:
            #     break
            ip_topic = tag
            to_write[ip_topic] = {}
            # we generate 10 poems for each tag with randomly chosen keywords.
            for k in range(10):
                # we will randomly sample 3 keywords
                random_kws = random.sample(kws, 5)
                kw = ','.join(random_kws)
                
                condition = SPECIAL_TOKENS['bos_token'] + ip_topic + \
                SPECIAL_TOKENS['sep_token'] + kw + SPECIAL_TOKENS['sep_token']
                # print(condition)
                
                generated = torch.tensor(tokenizer.encode(condition)).unsqueeze(0)
                generated = generated.to(DEVICE)
                
                # Top-k sampling
                # sample_outputs = model.generate(
                #                                 generated, 
                #                                 do_sample=True,   
                #                                 top_k=50, 
                #                                 max_length=500,
                #                                 top_p=0.95, 
                #                                 num_return_sequences=1
                #                                 )
                
                # Beam search
                sample_outputs = model.generate(
                                                generated, 
                                                do_sample=True,   
                                                num_beams=5, 
                                                max_length=500,
                                                num_return_sequences=1,
                                                no_repeat_ngram_size=2,
                                                early_stopping=True
                                                )
                
                f_name = "results/" + "gpt-2-topic_kw_cond_sample_poems_beam" + ".json"
                
                for i, sample_output in enumerate(sample_outputs):
                    text = tokenizer.decode(sample_output, skip_special_tokens=True)
                    a = len(ip_topic) + len(','.join(random_kws))
                    gen_poem = text[a:]
                    
                    to_write[ip_topic].update({
                        str(kw) : gen_poem
                        })
                    
        with open(f_name, 'w') as fp:
            json.dump(to_write, fp)
                    

class ComputePerplexity():
    # Compute PPL of generated poems
    @staticmethod
    def compute_ppl(tokenizer, params):
        
        # Load trained model
        configuration = AutoConfig.from_pretrained(MODEL,
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    sep_token_id=tokenizer.sep_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    output_hidden_states=False)
        
        model = AutoModelForPreTraining.from_pretrained(MODEL, config=configuration)
        model.resize_token_embeddings(len(tokenizer))

        model_path = glob.glob(params.out_dir + "/*.pth")[0]
        print("Using model: ", model_path)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.cuda()  
        
        # Load JSON file with generated poems
        with open(params.gen_poem_json) as json_file:
            data = json.load(json_file)
    
        to_write = {}
        k=0
        count = 0
        for actual_tag, line in data.items():
            k+= 1
            print("Progress (144 poems)- ", k, end="\r")
            to_write[actual_tag] = {}
            for id, (kw, poem) in enumerate(line.items()):
                if poem == "":
                    count += 0
                    pass
                else:
                    to_write[actual_tag][id] = {}
                    ppl = get_ppl(poem, model, tokenizer)
                    to_write[actual_tag][id].update({
                        "Keywords": kw,
                        "Perplexity": ppl.item(),
                        "Poem": poem
                    })
        print("Total empty poems:", count)
        f_name = "results/" + "gpt-2-poems-topk" + ".json"
        with open(f_name, 'w') as fp:
            json.dump(to_write, fp)

        
def get_ppl(poem, model, tokenizer):
    encodings = tokenizer(poem, return_tensors="pt")
    
    max_length = model.config.n_positions
    stride = 512
    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)
    try:
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    except RuntimeError:
        pdb.set_trace()
    return ppl
            


## When generating new poems, disable random sampling and make sure, all input keywords are
# contrained in the generated poems.


