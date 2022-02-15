import enum
import re
import json
import random
import string
import pickle
import operator
from collections import defaultdict
from torch import le

from tqdm import tqdm
from RAKE import Rake

import strings
from util import read_file


def process_lines(lines):
    example = ["[BOP]"]

    for line in lines:
        example += line.split(' ')
        example += ["[EOL]"]

    example += ["[EOP]"]

    return example


def process_poems():
    example_id = 0
    dataset = {}

    # Rake setup with stopword directory.
    # Rake is used for keywords extraction.
    stop_dir = f"{strings.STOPWORDS_LIST_PATH}"
    rake_object = Rake(stop_dir)

    # Get cleaned poems
    with open("clean_poems.json", "r+") as json_file:
        data = json.load(json_file)

    # Make multiple examples from every poem, where a single example is
    # a complete stanza consisting of 4 to 8 lines.
    # Additionally, extract up to 5 keywords for every example.
    # A full example consists of:
    #     1. topic of the poem
    #     2. keywords from the selected lines
    #     3. poem's lines
    for _, items in tqdm(data.items()):
        poem = items["poem"]
        lines = poem.split('\n')
        examples = []

        split_line = random.choice([3, 4, 5])
        examples.append(lines[:split_line])
        if len(lines) > 8:
            examples.append(lines[split_line:split_line+4])

        for example in examples:
            keywords = rake_object.run("\n".join(example))
            keywords = list({pair[0].split(' ')[0] for pair in keywords[:min(5, len(keywords))]})
            keywords = [keyword.split('\n')[0] for keyword in keywords]

            if len(keywords) < 2:
                continue

            instance = {"topic": items['topic'], "keywords": keywords, "example": process_lines(example)}
            dataset[f"example_{example_id}"] = instance
            example_id += 1
    
    return dataset


def store_dataset(dataset, path):
    with open(f"{path}", 'w+') as json_file:
        json.dump(dataset, json_file, indent=4)


def topics_generator(topics_list):
    random.shuffle(topics_list)

    for topic in topics_list:
        yield topic


def generate_set(dataset, topics_dist, is_train=True):
    result_set = {}
    num_examples = 6
    id_counter = 0

    for ind, topic in enumerate(topics_generator(list(topics_dist.keys())), 1):
        if is_train:
            selected_examples = topics_dist[topic]
        else:
            selected_examples = random.sample(topics_dist[topic], num_examples)

            if ind == 8:
                num_examples = 7

        for example in selected_examples:
            result_set[f"example_{id_counter}"] = dataset[example]
            id_counter += 1
            if not is_train:
                topics_dist[topic].remove(example)
    
    return result_set


def split_dataset(dataset):
    topics_dist = defaultdict(list)
    for key, data in dataset.items():
        topics_dist[data["topic"]].append(key)
    
    valid_set = generate_set(dataset, topics_dist, False)
    test_set = generate_set(dataset, topics_dist, False)
    train_set = generate_set(dataset, topics_dist)

    store_dataset(train_set, "train_set.json")
    store_dataset(valid_set, "valid_set.json")
    store_dataset(test_set, "test_set.json")

    print(f"Dataset size:   {len(dataset)}")
    print(f"Train set size: {len(train_set)}")
    print(f"Valid set size: {len(valid_set)}")
    print(f"Test  set size: {len(test_set)}")


if __name__ == "__main__":
    dataset = process_poems()
    store_dataset(dataset, "eng_dataset.json")

    split_dataset(dataset)
