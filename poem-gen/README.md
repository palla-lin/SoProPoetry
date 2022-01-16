# Introduction
This repository uses GPT-2 transformer model to automatically generate poem of a given theme/topic. We use [NeuralPoet](https://example.com) dataset to fine-tune the transformer model to generate poetry style text of a specific length. Planned modifications
1.  Generate poems that follows strict stanza format.
2.  Generated poem should be conditioned on certain keywords as entered by the user.

# Getting started
- Help: for instructions on how to run training script with appropriate arguments.
```
python main.py --help
usage: main.py  [-h] 
                [-out_dir OUT_DIR] 
                [-learning_rate LEARNING_RATE] 
                [-epochs EPOCHS] 
                [-max_len MAX_LEN] 
                [-batch_size BATCH_SIZE]
                [-save_model SAVE_MODEL]
                dataset_obj

positional arguments:
  dataset_obj           path to pickled clean dataset

optional arguments:
  -h, --help            show this help message and exit
  -out_dir OUT_DIR      path to trained model
  -learning_rate LEARNING_RATE
                        learning rate [default: 1e-4]
  -epochs EPOCHS        number of training epochs [default: 8]
  -max_len MAX_LEN      maximum sequence length [default: 1024]
  -batch_size BATCH_SIZE
                        batch size while training [default: 2]
  -save_model SAVE_MODEL
                        save model [default: True]
```

- Training and poem generation
```
python main.py ../data/neural-poet/cleaned_poem_tags_dict.pkl
```

# Training and evaluation log
```
Epoch 1 of 8
Average Training Loss: 0.7812812092233021. Epoch Training Time: 0:22:19
Average Validation Loss: 0.6985488197560817

Epoch 2 of 8
Average Training Loss: 0.6781221078296606. Epoch Training Time: 0:22:16
Average Validation Loss: 0.689365719777143

Epoch 3 of 8
Average Training Loss: 0.6411326588424874. Epoch Training Time: 0:22:17
Average Validation Loss: 0.691284994184711

Epoch 4 of 8
Average Training Loss: 0.6063146302462313. Epoch Training Time: 0:22:17
Average Validation Loss: 0.6916554246386862

Epoch 5 of 8
Average Training Loss: 0.5756404871595745. Epoch Training Time: 0:22:17
Average Validation Loss: 0.6962963547866139

Epoch 6 of 8
Average Training Loss: 0.5487534548329559. Epoch Training Time: 0:22:17
Average Validation Loss: 0.7036655368811168

Epoch 7 of 8
Average Training Loss: 0.526841444655496. Epoch Training Time: 0:22:17
Average Validation Loss: 0.7126045300907016

Epoch 8 of 8
Average Training Loss: 0.50854119013143. Epoch Training Time: 0:22:18
Average Validation Loss: 0.7196817226584517

Total Training Time: 3:12:19
```

# Results
Samples of generated poem can be found in `sample_poem.txt`