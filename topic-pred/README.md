# Multilabel multiclass topic classification

This repository is concerned with multilabel topic prediction for poems. We use poetry foundation dataset with a total of 13854 poems (of which 955 poems are unlabeled) labelled into 129 topics like romance, nature, health, friendship etc. A bidirectional LSTM model is trained to capture the context in poems and predcit topics for the unlabeled poems.

## Dataset
Poetry foundation dataset. [LINK](https://www.kaggle.com/tgdivy/poetry-foundation-poems)
- Total poems = 13854
- Total topics = 129

## Getting started
- Help: for instructions on how to download pre-trained embeddings and run training script with appropriate arguments.
    ```
    wget https://nlp.stanford.edu/data/glove.6B.zip
    ```
    ```
    python main.py --help
    usage: main.py  [-h] 
                    [-trained_model TRAINED_MODEL] 
                    [-high_level_tags HIGH_LEVEL_TAGS] 
                    [-learning_rate LEARNING_RATE] 
                    [-epochs EPOCHS]
                    [-dropout DROPOUT] 
                    [-embedding_size EMBEDDING_SIZE] 
                    [-max_seq_len MAX_SEQ_LEN] 
                    [-batch_size BATCH_SIZE]
                    [-save_model SAVE_MODEL] 
                    [-print_samples PRINT_SAMPLES]
                    dataset_obj 
                    emb_f 
                    out_dir 
                    model_dir

    positional arguments:
      dataset_obj           path to pickle dataset
      emb_f                 path to pre-trained embeddings
      out_dir               path to save processed poems, tags
      model_dir             path to save trained model

    optional arguments:
      -h, --help            show this help message and exit
      -trained_model TRAINED_MODEL
                            path to trained model
      -high_level_tags HIGH_LEVEL_TAGS
                            take only high level topics
      -learning_rate LEARNING_RATE
                            learning rate [default: 0.001]
      -epochs EPOCHS        number of training epochs [default: 10]
      -dropout DROPOUT      the probability for dropout [default: 0.25]
      -embedding_size EMBEDDING_SIZE
                            number of embedding dimension [default: 300]
      -max_seq_len MAX_SEQ_LEN
                            maximum sequence length [default: 100]
      -batch_size BATCH_SIZE
                            batch size while training [default: 64]
      -save_model SAVE_MODEL
                            save model [default: True]
      -print_samples PRINT_SAMPLES
                            number of poems to print predicted tags for

    ```
- Training and Evaluation
    - use only high level tags: `"love", "nature", "life", "romantic", "freedom", "culture", "suicide"`
    ```
    python main.py ../data/neural-poet/poem_dict.obj ../fasttext/wiki-news-300d-1M-subword.vec data/NeuralPoet/ cnn -high_level_tags True -epochs 50
    ```
    - use all 144 tags
    ```
    python main.py poem_dict.obj ../fasttext/wiki-news-300d-1M-subword.vec data/NeuralPoet/ cnn
    ```
## Results
Results can be found under `train_logs.txt`

