"""
    This file should contain relevant strings (e.g. paths) frequently used in the project.
"""

# Paths
ENG_POEMS_PATH  = "data/eng_poems_dict.obj"
ENG_CORPUS_PATH = "data/eng_dataset.json"
ENG_TRAIN_PATH = "data/train_set.json"
ENG_VALID_PATH = "data/valid_set.json"
ENG_TEST_PATH = "data/test_set.json"

TOKEN_TO_ID_PATH = "data/vocabs/token_to_id.json"
ID_TO_TOKEN_PATH = "data/vocabs/id_to_token.json"
TOKEN_TO_ID_GLOVE_PATH = "data/vocabs/token_to_id_glove.json"
ID_TO_TOKEN_GLOVE_PATH = "data/vocabs/id_to_token_glove.json"

GLOVE_EMBED_PATH = "data/glove_embed/glove_embeddings.npy"
STOPWORDS_LIST_PATH = "data/smart_stoplist.txt"
CONFIG_PATH = "configs/encdec_config.json"

# Special tokens (Encoder Decoder)
BOP = "[BOP]"
EOP = "[EOP]"
EOL = "[EOL]"
PAD = "[PAD]"
UNK = "[UNK]"