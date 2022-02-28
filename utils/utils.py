"""
Splits the dataset into training, test and validation parts
"""
import pickle
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


def prep_poem(string):
    max_lines = 5  # 4 line poem
    res = []

    short_poem = string.lower().split('\n')[:max_lines]

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
    poems = dict()
    poem_idx = 0

    for key in data.keys():
        for p in data[key]:
            poems[poem_idx] = [key, prep_poem(p)]
            poem_idx += 1
    return poems

