import torch
import re
import loader
import os

from argparse import ArgumentParser

from allennlp.commands.train import train_model
from allennlp.training.trainer import Params
from allennlp.common.util import import_submodules
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive

from loader import NorecReaderHierarchical, NorecReader_Flat

from sklearn.metrics import accuracy_score, f1_score

def main():
    parser = ArgumentParser()
    parser.add_argument('--lang', action='store')
    parser.add_argument('--config', action='store', default='configs/HAN.jsonnet')
    parser.add_argument('--save', action='store', default='experiments/models/HAN')
    parser.add_argument('--dataset', default="conllu.tar.gz")
    args = parser.parse_args()

    import_submodules("loader")
    import_submodules("models")


    test_path = "data/{0}/{1}/test".format(args.lang, args.dataset)

    model = load_archive(os.path.join(args.save, "model.tar.gz")).model
    if "HAN" in args.config or "hier" in args.config:
        reader = NorecReaderHierarchical()
    else:
        reader = NorecReader_Flat()

    p = Predictor(model, reader)

    header = "doc_id\tgold\tpred\tnum_sents\tnum_tokens\n"

    predictions = []
    gold_labels = []
    output_text = header

    for i in reader.read(test_path):
        metadata = i.fields['meta'].metadata

        try:
            pred = p.predict_instance(i)['prediction']
        # if there's an error always choose 1
        except:
            pred = 1
        predictions.append(pred)

        gold_label = i["rating"].label
        gold_labels.append(gold_label)

        output_text += "{}\t{}\t{}\t{}\t{}\n".format(metadata["doc_id"],
                                                     gold_label,
                                                     pred,
                                                     metadata["sentences"],
                                                     metadata["tokens"])


    acc = accuracy_score(gold_labels, predictions)
    f1 = f1_score(gold_labels, predictions, average="macro")

    print("Acc score: {0:.3f}\nF1 score: {1:.3f}\n".format(acc, f1))


    final_output = "Acc score: {0:.3f}\nF1 score: {1:.3f}\n\n".format(acc, f1)
    final_output += output_text

    with open(os.path.join(args.save, "results.txt"), "w") as outfile:
        outfile.write(final_output)

main()
