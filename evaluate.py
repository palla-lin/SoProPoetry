#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:  Sun Jan 16 11:30:35 AM CET 2022

"""
Given predicted topics from topic classifier system and 
ppl scores computed on each generated poem, this script computes 
avg keyword usage, mean ppl across all poems and topic-classification
or topic relevance score.
"""


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

def evaluate_enc_dec_poems(args):
    """Compute topic classifier score on enc-dec generated poems"""
    with open(args.json_data) as json_file:
        data = json.load(json_file)
    
    with open(args.pred_tags_pkl, 'rb') as fp:
        # Tuple of (actual_tags, greedy_predicted_tags, ktop_predicted_tags)
        pred_tags = pickle.load(fp)
    
    # pdb.set_trace()
    greddy_topic_clf_score = 0
    ktop_topic_clf_score = 0
    for idx, (example_id, line) in enumerate(data.items()):
        actual_tag = line['topic']
        greedy_poem_pred_topic = line['greedy_poem_pred_topic']
        ktop_poem_pred_topic = line['ktop_poem_pred_topic']
        
        # perform a sanity check
        if actual_tag == pred_tags[idx][0]: # this must be true, otherwise there is some error
            if actual_tag == pred_tags[idx][1]:
                greddy_topic_clf_score += 1
            if actual_tag == pred_tags[idx][2]:
                ktop_topic_clf_score += 1
        else:
            print("ERROR: Actual tag from ", str(args.json_data), " and ",\
                args.pred_tags_pkl, " differ.")
    # pdb.set_trace()
    print("Avg topic relevance score (greddy): %.3f" % (greddy_topic_clf_score/len(pred_tags)))
    print("Avg topic relevance score (topk): %.3f" % (ktop_topic_clf_score/len(pred_tags)))
    
    

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
            json.dump(aku_ppl_score, fp,  ensure_ascii=False, indent=4)
    
    
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
    # evaluate_enc_dec_poems(args)

def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_data", help="input generated poem json file")
    parser.add_argument("pred_tags_pkl", help="file path to pickled tags predicted by topic classifier")
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()