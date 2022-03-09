import argparse

def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_obj", help="file path to pickle dataset")
    parser.add_argument("emb_f", help="file path to pre-trained embeddings (fastext word2vec)")
    parser.add_argument("out_dir", help="dir path to save pre-processed poems, tags, emb-weights")
    parser.add_argument("model_dir", help="dir path to save trained model")
    parser.add_argument("-trained_model", default=None, type=str, help='path to trained model')
    parser.add_argument("-high_level_tags", default=False, type=bool, help='take only high level topics')
    parser.add_argument("-learning_rate", default=0.001, type=float, help='learning rate [default: 0.001]')
    parser.add_argument("-epochs", default=20, type=int, help='number of training epochs [default: 20]')
    parser.add_argument("-dropout", default=0.15, type=float, help='the probability for dropout [default: 0.15]')
    parser.add_argument("-embedding_size", default=300, type=int, help='number of embedding dimension [default: 300]')
    parser.add_argument("-max_seq_len", default=1000, type=int, help='maximum sequence length [default: 100]')
    parser.add_argument("-batch_size", default=32, type=int, help='batch size while training [default: 32]')
    parser.add_argument("-save_model", default=True, type=bool, help='save model  [default: True]')
    parser.add_argument("-print_samples", default=5, type=int, help='number of poems to print predicted tags for')
    args = parser.parse_args()
    return args
