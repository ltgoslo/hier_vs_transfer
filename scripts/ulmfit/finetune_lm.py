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

    args = parser.parse_args()

    # Import the training data
    dataset = os.path.join("data", args.lang, "conllu.tar.gz")
    print("Importing data from {0}".format(dataset))
    train, dev, test = import_sentiment_conllu(dataset)


    print("Putting data in Data Bunch")
    data_lm = TextLMDataBunch.from_df(train_df=train,
                                      valid_df=dev,
                                      path="")

    # Make sure that the vocabulary is the same as during general pretraining
    print("loading vocab from general pretrianing...")
    full_data = load_data("data/{0}".format(args.lang), "data_lm_export.pkl")
    data_lm.vocab = full_data.vocab


    print("saving data bunch")
    data_lm.save("data/{0}/finetune_data_lm_export.pkl".format(args.lang))

    # Set up language model
    print("Setting up model")
    learn = language_model_learner(data_lm,
                                   arch=AWD_LSTM,
                                   drop_mult=0.1,
                                   pretrained=False,
                                   model_dir="models/finetuned_lm")

    # Get best pretrained language model
    best_perplexity, epochs, model_file = get_best_run(weightdir="models",
                                                       lang=args.lang,
                                                       model="full")


    # Load pretrained model
    print("Loading best model: {0}".format(model_file))

    learn.load(os.path.abspath(model_file))

    # Fine-tune on domain
    print("fine-tuning LM")


    # Use slanted triangular learning rates for each layer group
    lr_scheduler = SlantedTriangularLR()
    lower_layer_lrs = lr_scheduler.calculate(args.ft_epochs, 4e-4)
    middle_layer_lrs = lr_scheduler.calculate(args.ft_epochs, 4e-3)
    top_layer_lrs = lr_scheduler.calculate(args.ft_epochs, 4e-2)
    output_layer_lrs = lr_scheduler.calculate(args.ft_epochs, 1e-1)
    schedule = [lower_layer_lrs,
                middle_layer_lrs,
                top_layer_lrs,
                output_layer_lrs
                ]

    best_val = 1000000000000

    for i in range(args.ft_epochs):
        # get the learning rates for each layer group at each epoch
        lrs = [x[i] for x in schedule]
        learn.fit_one_cycle(1,
                            max_lr=lrs,
                            #wd=(1e-4, 1e-4, 1e-1))
                            )
        perplex = np.exp(learn.validate()[0])
        if perplex < best_val:
            best_val = perplex
            learn.save_encoder("{0}-encoder-epochs:{1}-val_perplex:{2}".format(args.lang, i + 1, perplex))
            learn.save("{0}-full-epochs:{1}-val_perplex:{2}".format(args.lang, i + 1, perplex))
