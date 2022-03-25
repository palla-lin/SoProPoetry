"""
Splits the dataset into training, test and validation parts
"""
import math
import pickle
import string
import matplotlib.pyplot as plt
PATH = "data/poem_dict.obj"

def read_file(path):

    with open(path, 'rb') as f:
        data = pickle.load(f)
     # split
        dict_to_list = list(data.items())
        # print("dict_to_list",len(dict_to_list))

        train_instance_number = int(len(data) * 0.7)
        # print("train_instance_number",train_instance_number)

        train_input = dict_to_list[:train_instance_number]
        # print("train_input", len(train_input))
        train_input = dict(train_input)

        test_val = dict_to_list[len(train_input):]
        test_input = test_val[:int(len(test_val)/2)]
        # print("test_input", len(test_input))

        val_input = test_val[int(len(test_input)):]
        # print("val_input", len(val_input))
        val_input = dict(val_input)
        test_input = dict(test_input)

    return train_input,test_input,val_input


def spec_chars(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

        # get special characters
        chars = set()
        for topic, poems in data.items():
            for poem in poems:
                lines = poem.split('\n')

                for line in lines:
                    specials = [char for char in line if char not in string.ascii_letters]
                    chars.update(set(specials))
    return chars


def prep_poem(poemstring, special_chars):
    max_lines = 5  # 4 line poem
    res = []

    clean_poem = ""

    for char in poemstring:
        if char not in special_chars:
            clean_poem += char

    short_poem = clean_poem.lower().split('\n')[:max_lines]

    res = sum([i.split() for i in short_poem], ['<BOP>'])
    res.append('<EOP>')

    max_len = 50
    if len(res) > max_len:
        res = res[:max_len]
    elif len(res) < max_len:
        diff = max_len - len(res)
        for i in range(diff):
            res.append("<pad>")

    return res


def get_dataset(data):
    special_chars = spec_chars(PATH)

    poems = dict()
    poem_idx = 0

    for key in data.keys():
        for p in data[key]:
            poems[poem_idx] = [[key], prep_poem(p, special_chars)]
            poem_idx += 1
    return poems


def plot_loss(train_losses,valid_losses):

    plt.figure(figsize=(12, 6))

    plt.plot(range(len(train_losses)), train_losses, label="train")
    plt.plot(range(len(valid_losses)), valid_losses, label="valid")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel('Loss')
    plt.grid(True)

    plt.show()


def plot_ppl(train_losses,valid_losses):

    plt.figure(figsize=(4, 6))

    tr_ppl = [math.exp(i) for i in train_losses]
    val_ppl = [math.exp(i) for i in valid_losses]
    plt.plot(range(len(train_losses)), tr_ppl, label="train")
    plt.plot(range(len(valid_losses)), val_ppl, label="valid")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


