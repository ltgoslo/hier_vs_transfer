# import libraries
import fastai
from fastai import *
from fastai.text import *
from import_dataset import *
from utils import *

import pandas as pd
import numpy as np
import os
import argparse


def main():
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="no")
    parser.add_argument("--ft_epochs", default=50, type=int)
    parser.add_argument("--dataset", default="conllu.tar.gz")
    parser.add_argument("--save", default="models/classifier")

    args = parser.parse_args()

    # Import the training data
    dataset = os.path.join("data", args.lang, args.dataset)
    print("Importing data from {0}".format(dataset))
    train, dev, test = import_sentiment_conllu(dataset)


    print("Putting data in Data Bunch")
    data_cl = TextClasDataBunch.from_df(train_df=train,
                                        valid_df=dev,
                                        path="")

    # Make sure that the vocabulary is the same as during general pretraining
    print("loading vocab from general pretrianing...")
    full_data = load_data("data/{0}".format(args.lang), "data_lm_export.pkl")
    data_cl.vocab = full_data.vocab

    print("saving data bunch")
    data_cl.save("data/{0}/classification_data_lm_export.pkl".format(args.lang))

    # Set up language model
    print("Setting up model")
    learn = text_classifier_learner(data_cl,
                                    arch=AWD_LSTM,
                                    drop_mult=0.7,
                                    pretrained=False,
                                    model_dir=args.save)

    # Get best pretrained language model
    best_p, epochs, model_file = get_best_run(weightdir="models/finetuned_lm",
                                              lang=args.lang,
                                              model="encoder")


    # Load pretrained model
    print("Loading best fine-tuned model: {0}".format(model_file))
    learn.load_encoder(os.path.abspath(model_file))


    # Fine-tune classifier
    print("fine-tuning classifier")

    # Perform gradual unfreezing (discriminative learning from Howard and Ruder (2018))

    # only updating linear layer
    learn.freeze()
    learn.fit_one_cycle(cyc_len=5, max_lr=1e-3, moms=(0.8, 0.7))

    # updating last two layers
    learn.freeze_to(-2)
    learn.fit_one_cycle(5, slice(1e-4, 1e-2), moms=(0.8, 0.7))

    # updating last three layers
    learn.freeze_to(-3)
    learn.fit_one_cycle(5, slice(1e-5, 5e-3), moms=(0.8, 0.7))

    # updating everything
    learn.unfreeze()
    learn.fit_one_cycle(5, slice(1e-5, 1e-3), moms=(0.8, 0.7))

    learn.save("{0}-full".format(args.lang))

