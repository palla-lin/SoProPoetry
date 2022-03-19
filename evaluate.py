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


def keywords_usage(kws, poem):
    "compute what fraction of keywords were used in the generated poems"
    score_arr = np.zeros(5)
    for id, kw in enumerate(kws):
        if kw in poem:
            # kewyord found
            score_arr[id] = 1
    score = np.mean(score_arr)
    return score
    

def load_data(args):
    with open(args.json_data) as json_file:
        data = json.load(json_file)
    
    with open(args.pred_tags_pkl, 'rb') as fp:
        # Tuple of (actual_tags, predicted_tags)
        pred_tags = pickle.load(fp)
    
    # Average-keyword-usage-ppl-score
    aku_ppl_score = {}
    iter = -1
    for actual_tag, line in data.items():
        score = 0
        ppl = 0
        topic_clf_score = 0
        # pdb.set_trace()
        for id, (kws, ppl_score, poem) in line.items():
            kws = line[id][kws].split(",")
            # pdb.set_trace()
            text = line[id][poem]
            score += keywords_usage(kws, text)
            ppl += line[str(id)][ppl_score]
            
            iter += 1
            if actual_tag == pred_tags[iter][0]: # Sanity check, match found
                if pred_tags[iter][-1] == actual_tag:
                    topic_clf_score += 1
                # iter += 1 # increase the count only when a perfect match is found.
                    
            else:
                pdb.set_trace()
                print("ERROR: Actual tag from ", str(args.json_data), " and ",\
                    args.pred_tags_pkl, " differ.")
            
            
        aku_ppl_score[actual_tag] = {
            "avg_kw_usage_score":  float("{:.3f}".format(score / len(line))),
            "avg_ppl_score": float("{:.4f}".format(ppl / len(line))),
            "avg_topic_clf_score": float("{:.3f}".format(topic_clf_score/ len(line)))   # avg over 10 poems
        }
    with open('poem-gen/results/final-results-topk.json', 'w') as fp:
            json.dump(aku_ppl_score, fp)
    
    
    net_score = np.zeros((len(aku_ppl_score),3))
    for id, (tag, score) in enumerate(aku_ppl_score.items()):
        
        net_score[int(id)][0] = score["avg_kw_usage_score"]
        net_score[int(id)][1] = score["avg_ppl_score"]
        net_score[int(id)][2] = score["avg_topic_clf_score"]
    
    print("Avergae keyword usage across all 144 topics: %.3f" % np.mean(net_score, axis=0)[0])
    print("Avergae PPL score across all 144 topics: %.3f" % np.mean(net_score,axis=0)[1])
    print("Avergae topic classifier score \
        (topics predicted by a trained topic classifier system) across all 144 topics: %.3f" % np.mean(net_score,axis=0)[2])

def main():
    """ main method """
    args = parse_arguments()
    load_data(args)

def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_data", help="input generated poem json file")
    parser.add_argument("pred_tags_pkl", help="file path to pickled tags predicted by topic classifier")
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()