import sys
import os
import argparse
import csv
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

from import_dataset import *


def get_best_C(Xtrain, ytrain,
               Xdev, ydev):
    """
    Find the best parameters on the dev set.
    """
    best_f1 = 0
    best_c = 0

    labels = sorted(set(ytrain))

    test_cs = [0.001, 0.003, 0.006, 0.009,
                   0.01,  0.03,  0.06,  0.09,
                   0.1,   0.3,   0.6,   0.9,
                   1,       3,    6,     9,
                   10,      30,   60,    90]
    for i, c in enumerate(test_cs):

        sys.stdout.write('\rRunning cross-validation: {0} of {1}'.format(i+1, len(test_cs)))
        sys.stdout.flush()

        clf = LinearSVC(C=c)
        h = clf.fit(Xtrain, ytrain)
        pred = clf.predict(Xdev)

        dev_f1 = f1_score(ydev, pred, labels=labels, average="micro")
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_c = c

    print()
    print('Best F1 on dev data: {0:.3f}'.format(best_f1))
    print('Best C on dev data: {0}'.format(best_c))

    return best_c, best_f1

def dummy(doc):
    return doc




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='test embeddings on a suite of datasets')
    parser.add_argument('--lang', help='output file for results', default='no')
    parser.add_argument('--dataset', default="conllu.tar.gz")

    args = parser.parse_args()

    # Info
    print(args)

    # Get the data from csv file
    print("opening data...")
    train, dev, test = import_sentiment_conllu(os.path.join("data",
                                                            args.lang,
                                                            args.dataset))

    vectorizer = CountVectorizer(tokenizer=dummy, preprocessor=dummy)
    X = vectorizer.fit(train["text"])


    # Convert labels to ints and text to bag of words
    print("converting to bow reps...")
    train_x = vectorizer.transform(train["text"])
    dev_x = vectorizer.transform(dev["text"])
    test_x = vectorizer.transform(test["text"])

    train_y = train["label"]
    dev_y = dev["label"]
    test_y = test["label"]


    # Give info about data
    labels = sorted(set(test_y))
    training_examples, voc_size = train_x.shape

    print()
    print("Labels: {0}".format(labels))
    print("Num training examples: {0}".format(training_examples))
    print("Vocabulary Size: {0}".format(voc_size))
    print()

    # Train SVM
    print("tuning C parameter")
    best_c, best_f1 = get_best_C(train_x, train_y,
                                 dev_x, dev_y)

    print("training model...")
    clf = LinearSVC(C=best_c)
    clf.fit(train_x, train_y)

    # Test SVM
    preds = clf.predict(test_x)
    f1 = f1_score(test_y, preds, average="macro")
    micro_f1 = f1_score(test_y, preds, average="micro")
    acc = accuracy_score(test_y, preds)

    print("Acc: {0:.3f}".format(acc))
    print("Micro F1: {0:.3f}".format(micro_f1))
    print("Macro F1: {0:.3f}".format(f1))

    output_text = header = "doc_id\tgold\tpred\tnum_sents\tnum_tokens\n"

    for idx, gold, pred, num_sents, num_tokens in zip(test["doc_ids"],
                                                      test_y,
                                                      preds,
                                                      test["num_sents"],
                                                      test["num_tokens"]):
        output_text += "{}\t{}\t{}\t{}\t{}\n".format(idx,
                                                     gold,
                                                     pred,
                                                     num_sents,
                                                     num_tokens)

    final_output = "Acc score: {0:.3f}\nF1 score: {1:.3f}\n\n".format(acc, f1)
    final_output += output_text

    basedir = "experiments/models/bow/{0}".format(args.lang)
    os.makedirs(basedir, exist_ok=True)

    with open(os.path.join(basedir, "results.txt"), "w") as outfile:
        outfile.write(final_output)
