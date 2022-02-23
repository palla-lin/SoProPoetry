import math
import json
import random
from unittest import result

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.seq2seq import EncoderDecoder
from utils.util import read_file, set_seed, pad_and_sort_batch
from utils.strings import CONFIG_PATH, ENG_TEST_PATH, PAD
from data_loaders.eng_dataset import EnglishPoetryDataset


def test(config):
    test_set = EnglishPoetryDataset(ENG_TEST_PATH, config["context_size"], config["poem_size"], config["use_glove"])
    data_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=pad_and_sort_batch)
    pad_idx = test_set.get_token_id(PAD)

    device = torch.device('cuda' if config["use_cuda"] else 'cpu')
    
    model_name = config["model_name"]
    model = EncoderDecoder(config, test_set, pad_idx, device).to(device)
    model.load_state_dict(torch.load(model_name))

    criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

    model.eval()
    test_loss = 0
    total_num_keywords = 0
    num_used_keywords_greedy = 0
    num_used_keywords_topk = 0
    gen_poems_dict = {}
    example_id = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            contexts = batch[0].cuda()
            poems = batch[1].cuda()
            context_lens = batch[2].cuda()

            output = model(contexts, context_lens, poems)
            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            poems = poems.permute(1, 0)
            poems = poems[1:].contiguous().view(-1)

            loss = criterion(output, poems)
            test_loss += loss.item()

            keywords = [test_set.get_ids_token(str(word_id.item())) for word_id in contexts[0].cpu()]
            gen_poem_greedy = [test_set.get_ids_token(str(token_id.item())) for token_id in output.argmax(1)]
            gen_poem_greedy = " ".join(gen_poem_greedy)
            gen_poem_ktop = [test_set.get_ids_token(str(random.choice(tokens_ids).item())) for tokens_ids in torch.topk(output, 5, dim=1).indices]
            gen_poem_ktop = " ".join(gen_poem_ktop)

            curr_correct_keywords_greedy = 0
            curr_correct_keywords_topk = 0
            for keyword in keywords:
                if keyword in gen_poem_greedy:
                    curr_correct_keywords_greedy += 1
                if keyword in gen_poem_ktop:
                    curr_correct_keywords_topk += 1
            total_num_keywords += len(keywords)
            num_used_keywords_greedy += curr_correct_keywords_greedy
            num_used_keywords_topk += curr_correct_keywords_topk

            result = {"keywords": keywords, 
                      "greedy_poem": gen_poem_greedy,
                      "greedy_kw_used": f"{curr_correct_keywords_greedy}/{len(keywords)}",
                      "ktop_poem": gen_poem_ktop,
                      "ktop_kw_used": f"{curr_correct_keywords_topk}/{len(keywords)}"}

            gen_poems_dict[f"example_{example_id}"] = result
            example_id += 1
    
    test_loss /= len(data_loader)
    greedy_kw_used = num_used_keywords_greedy / total_num_keywords
    ktop_kw_used = num_used_keywords_topk / total_num_keywords

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    print(f'| Greedy keywords used: {greedy_kw_used:.3f} | Ktop keywords used: {ktop_kw_used:7.3f} |')
    
    with open(f"{model_name.split('.')[0]}_poems.json", 'w+') as json_file:
        json.dump(gen_poems_dict, json_file, indent=4)


if __name__ == "__main__":
    model_config = read_file(CONFIG_PATH)
    set_seed(model_config["seed"])

    test(model_config)
