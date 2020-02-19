import lzma
import numpy as np
import pandas as pd
import glob
import os
from fastai.text import TextLMDataBunch
from sklearn.model_selection import train_test_split


def import_lm_data_conllu(corpus):

    num_tokens = 0
    sentences = []

    # for each file in the corpus
    file = lzma.open(corpus)
    tokens = []

    # go through each line in conllu format
    for line in file:
        line = line.decode("utf8")
        # for tokens, the lines start with token id
        if line[0].isnumeric():
            token = line.strip().split("\t")[1]
            tokens.append(token)
            num_tokens += 1
        elif line == "\n":
            sentences.append(tokens)
            tokens = []

    # put everything in pandas dataframes
    df = pd.DataFrame({"label": np.zeros(len(sentences)), "text": sentences})
    train_df, dev_df = train_test_split(df,
                                        test_size=0.05,
                                        random_state=12)

    print("Total number of tokens: {0}".format(num_tokens))
    print("Train: {0}".format(int(num_tokens * .95)))
    print("Dev: {0}".format(int(num_tokens * .05)))

    return train_df, dev_df
