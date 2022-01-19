from torch.utils.data import DataLoader

from data_loaders.eng_dataset import EnglishPoetryDataset


dataset = EnglishPoetryDataset()
data_loader = DataLoader(dataset, batch_size=5, shuffle=True)

for batch in data_loader:
    print(batch)
    break