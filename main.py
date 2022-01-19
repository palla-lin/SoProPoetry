import torch
from torch.utils.data import DataLoader

from data_loaders.eng_dataset import EnglishPoetryDataset
from utils.strings import CONFIG_PATH
from utils.util import read_file


dataset = EnglishPoetryDataset()
data_loader = DataLoader(dataset, batch_size=5, shuffle=True)
model_config = read_file(CONFIG_PATH)

for batch in data_loader:
    print(torch.stack(batch[1]))
    break