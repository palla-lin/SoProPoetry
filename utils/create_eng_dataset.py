import re
import json
import string
import pickle
import operator

from tqdm import tqdm
from RAKE import Rake

from strings import ENG_CORPUS_PATH, ENG_POEMS_PATH, STOPWORDS_LIST_PATH


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

    # Reka setup with stopword directory.
    # Reka is used for keywords extraction.
    stop_dir = f"../{STOPWORDS_LIST_PATH}"
    rake_object = Rake(stop_dir)

    # Open raw data.
    with open(f"../{ENG_POEMS_PATH}", 'rb') as pickle_file:
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
            lines = poem.split('\n')[:4]

            keywords = rake_object.run("\n".join(lines))
            keywords = list({process_word(pair[0].split(' ')[0]) for pair in keywords[:min(5, len(keywords))]})

            instance = {"topic": topic, "keywords": keywords, "example": process_lines(lines)}
            dataset[f"example_{example_id}"] = instance

            example_id += 1
    
    return dataset


def store_dataset(dataset):
    with open(f"../{ENG_CORPUS_PATH}", 'w+') as json_file:
        json.dump(dataset, json_file, indent=4)


if __name__ == "__main__":
    dataset = process_poems()

    store_dataset(dataset)
