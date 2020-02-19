import os
import re


class SlantedTriangularLR():
    """
    Implementation of the slanted triangular learning rate scheduler
    """
    def __init__(self,
                 cut_frac=0.1,
                 ratio=32,
                 ):
        self.cut_frac = cut_frac
        self.ratio = ratio

    def calculate(self, T, lr_max=0.01):
        cut = int(T * self.cut_frac)

        lrs = []
        for t in range(T):
            if t < cut:
                p = t / cut
            else:
                p = 1 - ((t - cut) / (cut * (1 / self.cut_frac - 1)))

            lr = lr_max * ((1 + p * (self.ratio - 1)) / self.ratio)
            lrs.append(lr)
        return lrs

def get_best_run(weightdir, lang="no", model="full"):
    """
    This returns the best perplex, parameters, and weights from the models
    found in the weightdir. The "model" parameter can be either "full" or "encoder".
    """

    best_params = []
    best_perplexity = 10000.0
    best_weights = ''

    for file in os.listdir(weightdir):
        try:
            mlang, mmodel, epochs, perplexity = file.split('-')
            if mlang == lang and mmodel == model:
                epochs = int(epochs.split(":")[-1])
                perplexity = float(re.findall("[0-9]+\.[0-9]+", perplexity.split(":")[-1])[0])
                #print(perplexity)
                #print(type(perplexity))
                if perplexity < best_perplexity:
                    best_params = [epochs]
                    best_perplexity = perplexity
                    weights = os.path.join(weightdir, file)
                    best_weights = weights
        except ValueError:
            pass
    return best_perplexity, best_params, best_weights[:-4]
