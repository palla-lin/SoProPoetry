import math
import json

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.seq2seq import EncoderDecoder
from utils.util import read_file, set_seed, pad_and_sort_batch
from utils.strings import CONFIG_PATH, ENG_TEST_PATH, PAD
from data_loaders.eng_dataset import EnglishPoetryDataset


def test(config):
    test_set = EnglishPoetryDataset(ENG_TEST_PATH, config["context_size"], config["poem_size"])
    data_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=pad_and_sort_batch)
    pad_idx = test_set.get_token_id(PAD)

    device = torch.device('cuda' if config["use_cuda"] else 'cpu')
    
    model_name = config["model_name"]
    model = EncoderDecoder(config, pad_idx, device).to(device)
    model.load_state_dict(torch.load(model_name))

    criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

    model.eval()
    test_loss = 0
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

            keywords = [test_set.get_ids_token(int(word_id)) for word_id in contexts[0].cpu()]
            gen_poem = [test_set.get_ids_token(int(token_id)) for token_id in output.argmax(1)]
            gen_poem = " ".join(gen_poem)

            gen_poems_dict[f"example_{example_id}"] = {"keywords": keywords, "poem": gen_poem}
            example_id += 1
    
    test_loss /= len(data_loader)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    
    with open(f"{model_name.split('.')[0]}_poems.json", 'w+') as json_file:
        json.dump(gen_poems_dict, json_file, indent=4)


if __name__ == "__main__":
    model_config = read_file(CONFIG_PATH)
    set_seed(model_config["seed"])

    test(model_config)
