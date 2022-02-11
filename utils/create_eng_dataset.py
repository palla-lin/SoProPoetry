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


ALPHABET = list(string.ascii_letters)
LINES_LIMIT = 4


def process_word(token):
    if not re.search('[a-zA-Z]', token):
        return None

    while token[0] not in ALPHABET:
        token = token[1:]
    while token[-1] not in ALPHABET:
        token = token[:-1]
    
    return token.lower().strip()


def process_lines(lines):
    example = ["[BOP]"]

    for line in lines:
        for word in line.split(' '):
            word = process_word(word)
            if word:
                example += [word]
        example += ["[EOL]"]

    example[-1] = "[EOP]"

    return example


def process_poems():
    example_id = 0
    dataset = {}

    # Rake setup with stopword directory.
    # Rake is used for keywords extraction.
    stop_dir = f"{strings.STOPWORDS_LIST_PATH}"
    rake_object = Rake(stop_dir)

    # Open raw data.
    with open(f"{strings.ENG_POEMS_PATH}", 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    # Make multiple examples from every poem, where a single example is
    # a complete stanza consisting of 4 to 8 lines.
    # Additionally, extract up to 5 keywords for every example.
    # A full example consists of:
    #     1. poem's lines
    #     2. topic of the poem
    #     3. keywords from the selected lines
    for topic, poems in tqdm(data.items()):
        for poem in poems:
            lines = poem.split('\n')[:LINES_LIMIT]

            keywords = rake_object.run("\n".join(lines))
            keywords = list({process_word(pair[0].split(' ')[0]) for pair in keywords[:min(5, len(keywords))]})
            
            # Exclude bad examples
            if len(keywords) < 2:
                continue

            example = process_lines(lines)
            instance = {"topic": topic, "keywords": keywords, "example": example}
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
    
    validation_set = generate_set(dataset, topics_dist, False)
    test_set = generate_set(dataset, topics_dist, False)
    train_set = generate_set(dataset, topics_dist)

    store_dataset(train_set, strings.ENG_TRAIN_PATH)
    store_dataset(validation_set, strings.ENG_VALID_PATH)
    store_dataset(test_set, strings.ENG_TEST_PATH)


if __name__ == "__main__":
    dataset = process_poems()

    store_dataset(dataset, strings.ENG_CORPUS_PATH)

    split_dataset(dataset)

    train_set = read_file(strings.ENG_TRAIN_PATH)
    valid_set = read_file(strings.ENG_VALID_PATH)
    test_set = read_file(strings.ENG_TEST_PATH)
    corpus = read_file(strings.ENG_CORPUS_PATH)

    print(f"Train set size: {len(train_set)}")
    print(f"Valid set size: {len(valid_set)}")
    print(f"Test  set size: {len(test_set)}")
    print(f"Corpuss size: {len(corpus)}")
