# import libraries
import fastai
from fastai import *
from fastai.text import *
from import_dataset import *
from utils import *

from sklearn.metrics import accuracy_score, f1_score

import pandas as pd
import numpy as np
import os
import argparse


def main():
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="no")
    parser.add_argument('--save', action='store', default='experiments/models/ulmfit')
    parser.add_argument("--dataset", default="conllu.tar.gz")

    args = parser.parse_args()

    # Import the training data
    dataset = os.path.join("data", args.lang, args.dataset)
    print("Importing data from {0}".format(dataset))
    train, dev, test = import_sentiment_conllu(dataset)


    print("Putting data in Data Bunch")
    data_cl = load_data("data/{0}".format(args.lang), "classification_data_lm_export.pkl")

    # Make sure that the vocabulary is the same as during general pretraining
    print("loading vocab from general pretraining...")
    full_data = load_data("data/{0}".format(args.lang), "data_lm_export.pkl")
    data_cl.vocab = full_data.vocab

    # Set up language model
    print("Setting up model")
    learn = text_classifier_learner(data_cl,
                                    arch=AWD_LSTM,
                                    drop_mult=0.7,
                                    pretrained=False,
                                    model_dir="models/classifier")

    # Get pretrained classifier
    model_file = os.path.join("models", "classifier", "{0}-full".format(args.lang))


    # Load pretrained model
    print("Loading best fine-tuned model: {0}".format(model_file))
    learn.load(os.path.abspath(model_file))

    learn.data.add_test(test["text"])
    preds, y = learn.get_preds(ds_type=DatasetType.Test)

    acc = accuracy_score(preds.argmax(1)+1, test["label"])
    f1 = f1_score(preds.argmax(1)+1, test["label"], average="macro")

    final_output = "Acc score: {0:.3f}\nF1 score: {1:.3f}\n\n".format(acc, f1)

    final_output += "doc_id\tgold\tpred\tnum_sents\tnum_tokens\n"

    for doc_id, gold, pred, num_sents, num_tokens in zip(test["doc_ids"], test["label"], preds.argmax(1)+1, test["num_sents"], test["num_tokens"]):
        final_output += "{}\t{}\t{}\t{}\t{}\n".format(doc_id,
                                                      gold,
                                                      pred,
                                                      num_sents,
                                                      num_tokens)

    # If the ulmfit dir doesn't exist, make it
    basedir = os.path.join(args.save, args.lang)
    os.makedirs(basedir, exist_ok=True)

    with open(os.path.join(args.save, args.lang, "results.txt"), "w") as outfile:
        outfile.write(final_output)



