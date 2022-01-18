import argparse

def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_obj", help="path to pickled clean dataset")
    
    parser.add_argument("-out_dir", default="gpt-2", help='path to trained model')
    parser.add_argument("-learning_rate", default=1e-4, type=float, help='learning rate [default: 1e-4]')
    parser.add_argument("-epochs", default=8, type=int, help='number of training epochs [default: 8]')
    parser.add_argument("-max_len", default=1024, type=int, help='maximum sequence length [default: 1024]')
    parser.add_argument("-batch_size", default=2, type=int, help='batch size while training [default: 2]')
    parser.add_argument("-save_model", default=True, type=bool, help='save model  [default: True]')
    args = parser.parse_args()
    return args