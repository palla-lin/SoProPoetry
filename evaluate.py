#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:  Sun Jan 16 11:30:35 AM CET 2022

import os
import sys
import argparse
import numpy as np
import pdb
import json
import pickle



def perplexity_score(arg_1, arg_2, args):
    """ purpose of my function """
    

def keywords_usage(kws, poem):
    "compute what fraction of keywords were used in the generated poems"
    score_arr = np.zeros(5)
    for id, kw in enumerate(kws):
        if kw in poem:
            # kewyord found
            score_arr[id] = 1
    score = np.mean(score_arr)
    return score
    

def topic_classifier_prediction(actual_tag, poem):
    "Given a poem, predict its topic using topic classifier trained"
    

def load_data(args):
    with open(args.json_data) as json_file:
        data = json.load(json_file)
    
    with open(args.pred_tags_pkl, 'rb') as fp:
            actual_pred_tags = pickle.load(fp)
    
    pdb.set_trace()
    # Average-keyword-usage-ppl-score
    aku_ppl_score = {}
    for actual_tag, line in data.items():
        score = 0
        ppl = 0
        # pdb.set_trace()
        for id, (kws, ppl_score, poem) in line.items():
            kws = line[id][kws].split(",")
            # pdb.set_trace()
            text = line[id][poem]
            score += keywords_usage(kws, text)
            ppl += line[str(id)][ppl_score]
            
        aku_ppl_score[actual_tag] = {
            "avg_kw_usage_score": float(score / len(line)),
            "avg_ppl_score": float(ppl / len(line))
        }
    with open('poem-gen/results/final-results-beam.json', 'w') as fp:
            json.dump(aku_ppl_score, fp)
    
    
    net_score = np.zeros((len(aku_ppl_score),2))
    for id, (tag, score) in enumerate(aku_ppl_score.items()):
        
        net_score[int(id)][0] = score["avg_kw_usage_score"]
        net_score[int(id)][1] = score["avg_ppl_score"]
    
    print("Avergae keyword usage across all 144 topics: %.3f" % np.mean(net_score, axis=0)[0])
    print("Avergae PPL score across all 144 topics: %.3f" % np.mean(net_score,axis=0)[1])


def main():
    """ main method """
    args = parse_arguments()
    # os.makedirs(args.out_dir, exist_ok=True)
    load_data(args)


def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_data", help="input json file")
    parser.add_argument("pred_tags_pkl", help="file path to pickled pred tags by topic classifier")
    # parser.add_argument("-optional_arg", default=default_value, type=int, help='optional_arg meant for some purpose')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()