import overrides
import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional, Tuple
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.activations import Activation
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.attention import Attention
from allennlp.modules.feedforward import FeedForward
from allennlp.nn.util import get_text_field_mask, get_final_encoder_states

from allennlp.models import Model


@Model.register('hierarchical_doc')
class HierarchicalDoc(Model):
    def __init__(self, vocab, embed_dim: int,
                 word_encoder: Seq2SeqEncoder,
                 sent_encoder: Seq2SeqEncoder,
                 word_attn: Attention,
                 sent_attn: Attention):
        super().__init__(vocab)

        self._vocab = vocab
        self._embed = Embedding(self._vocab.get_vocab_size('tokens'), embed_dim)
        self._word_rnn = word_encoder
        self._sent_rnn = sent_encoder

        word_output_dim = self._word_rnn.get_output_dim()
        sent_output_dim = self._sent_rnn.get_output_dim()
        self._word_proj = FeedForward(word_output_dim, 1, word_output_dim, nn.Tanh())
        self._word_rand = nn.Parameter(torch.rand(word_output_dim))
        self._word_attn = word_attn
        self._sent_proj = FeedForward(sent_output_dim, 1, sent_output_dim, nn.Tanh())
        self._sent_rand = nn.Parameter(torch.rand(sent_output_dim))
        self._sent_attn = sent_attn

        self._doc_project = nn.Linear(sent_output_dim, self._vocab.get_vocab_size('labels'))
        self._crit = nn.CrossEntropyLoss()
        self._acc = CategoricalAccuracy()


    def forward(self, doc, rating, meta):
        output = {}
        mask = get_text_field_mask(doc, num_wrapping_dims=1)
        doc = self._embed(doc['tokens'])

        batch_size, num_sents, num_words = mask.size()
        word_reps = doc.view(batch_size * num_sents, num_words, -1)
        word_mask = mask.view(batch_size * num_sents, num_words)

        # calculate attention with projected reps, but weight the unprojected ones
        word_reps = self._word_rnn(word_reps, word_mask)
        projected_word_reps = self._word_proj(word_reps)
        word_probs = self._word_attn(self._word_rand.repeat(batch_size * num_sents, 1),
                                     projected_word_reps, word_mask)

        # first mask for a word will be 1 if the entire sentence isn't masked
        sent_reps = (word_probs.unsqueeze(-1) * word_reps).sum(dim=-2).view(batch_size, num_sents, -1)
        sent_mask = mask[:,:,0]
        sent_reps = self._sent_rnn(sent_reps, sent_mask)
        projected_sent_reps = self._sent_proj(sent_reps)
        sent_probs = self._sent_attn(self._sent_rand.repeat(batch_size, 1),
                                     projected_sent_reps, sent_mask)

        passage_reps = (sent_probs.unsqueeze(-1) * sent_reps).sum(dim=-2)
        clf = self._doc_project(passage_reps)

        output['prediction'] = [self._vocab.get_token_from_index(i.item(), 'labels') for i in clf.argmax(dim=-1)]
        output['loss'] = self._crit(clf, rating)
        self._acc(clf, rating)

        return output

    def get_metrics(self, reset=False):
        if not self.training:
            return {'acc': self._acc.get_metric(reset)}
        else:
            return {}

@Model.register('hierarchical_cnn')
class HierarchicalCNN(Model):
    def __init__(self, vocab, embed_dim: int,
                 word_encoder: Seq2VecEncoder,
                 sent_encoder: Seq2VecEncoder):
        super().__init__(vocab)

        self._vocab = vocab
        self._embed = Embedding(self._vocab.get_vocab_size('tokens'), embed_dim)
        self._word_cnn = word_encoder
        self._sent_cnn = sent_encoder

        word_output_dim = self._word_cnn.get_output_dim()
        sent_output_dim = self._sent_cnn.get_output_dim()

        self._doc_project = nn.Linear(sent_output_dim, self._vocab.get_vocab_size('labels'))
        self._crit = nn.CrossEntropyLoss()
        self._acc = CategoricalAccuracy()

    def forward(self, doc, rating, meta):
        output = {}

        # extra padding
        batch_size, num_sents, num_words = doc['tokens'].shape
        largest_filter = max([i.kernel_size[0] for i in self._sent_cnn._convolution_layers])
        if num_sents < largest_filter:
            z = torch.fill_(torch.zeros(batch_size, largest_filter - num_sents, num_words, dtype=torch.long),
                            self.vocab.get_token_index('@@PADDING@@'))
            doc['tokens'] = torch.cat((doc['tokens'], z), dim=1)

        mask = get_text_field_mask(doc, num_wrapping_dims=1)
        doc = self._embed(doc['tokens'])

        batch_size, num_sents, num_words = mask.size()
        word_reps = doc.view(batch_size * num_sents, num_words, -1)
        word_mask = mask.view(batch_size * num_sents, num_words)
        sent_mask = mask[:, :, 0]

        sent_reps = self._word_cnn(word_reps, word_mask).view(batch_size, num_sents, -1)
        #print(sent_reps.shape)
        #print(sent_mask.shape)
        passage_reps = self._sent_cnn(sent_reps, sent_mask)

        clf = self._doc_project(passage_reps)

        output['prediction'] = [self._vocab.get_token_from_index(i.item(), 'labels') for i in clf.argmax(dim=-1)]
        output['loss'] = self._crit(clf, rating)
        self._acc(clf, rating)

        return output

    def get_metrics(self, reset=False):
        if not self.training:
            return {'acc': self._acc.get_metric(reset)}
        else:
            return {}


# @Model.register("mycnn")
# class MyCNN(Model):
#     def __init__(self,
#                  vocab,
#                  num_filters,
#                  kernels,
#                  dropout,
#                  embedding_dim):
#         super().__init__(vocab)
#         self._vocab = vocab
#         self.num_filters = num_filters
#         self.kernels = kernels
#         self.vocab_size = vocab.get_vocab_size()
#         self.embedding_length = embedding_dim

#         self.word_embeddings = nn.Embedding(self.vocab_size, embedding_dim)
#         self.convs = []
#         for kernel in kernels:
#             self.convs.append(nn.Conv2d(1,
#                                         num_filters,
#                             (kernel, embedding_dim)))
#         self.dropout = nn.Dropout(dropout)
#         self.label = nn.Linear(len(kernels)*num_filters, self._vocab.get_vocab_size('labels'))

#         self._crit = nn.CrossEntropyLoss()
#         self._acc = CategoricalAccuracy()

#         if torch.cuda.is_available():
#             self.word_embeddings.cuda()
#             for conv in self.convs:
#                 conv.cuda()
#             self.dropout.cuda()
#             self.label.cuda()




#     def conv_block(self, input, conv_layer):
#         conv_out = conv_layer(input)
#         # conv_out.size() = (batch_size, out_channels, dim, 1)
#         activation = F.relu(conv_out.squeeze(3))
#         # activation.size() = (batch_size, out_channels, dim1)
#         max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
#         # maxpool_out.size() = (batch_size, out_channels)
#         return max_out

#     def forward(self, tokens, rating, meta):
#         #print(tokens)
#         input = self.word_embeddings(tokens["tokens"])
#         # input.size() = (batch_size, num_seq, embedding_length)
#         input = input.unsqueeze(1)
#         # input.size() = (batch_size, 1, num_seq, embedding_length)
#         max_out = []
#         for conv in self.convs:
#             max_out.append(self.conv_block(input, conv))
#         all_out = torch.cat(max_out, 1)
#         # all_out.size() = (batch_size, num_kernels*out_channels)
#         fc_in = self.dropout(all_out)
#         # fc_in.size()) = (batch_size, num_kernels*out_channels)
#         clf = self.label(fc_in)

#         output = {}
#         output['prediction'] = [self._vocab.get_token_from_index(i.item(), 'labels') for i in clf.argmax(dim=-1)]
#         output['loss'] = self._crit(clf, rating)
#         output['accuracy'] = self._acc(clf, rating)

#         return output

#     def get_metrics(self, reset=False):
#         if not self.training:
#             return {'accuracy': self._acc.get_metric(reset)}
#         else:
#             return {}

@Model.register('mycnn')
class HierarchicalCNN(Model):
    def __init__(self,
                 vocab,
                 embed_dim: int,
                 word_encoder: Seq2VecEncoder,
                 dropout: float = 0.3):
        super().__init__(vocab)

        self._vocab = vocab
        self._embed = Embedding(self._vocab.get_vocab_size('tokens'), embed_dim)
        self._word_cnn = word_encoder
        self.dropout = nn.Dropout(dropout)

        word_output_dim = self._word_cnn.get_output_dim()

        self._doc_project = nn.Linear(word_output_dim, self._vocab.get_vocab_size('labels'))
        self._crit = nn.CrossEntropyLoss()
        self._acc = CategoricalAccuracy()

    def forward(self, tokens, rating, meta):
        output = {}

        # extra padding
        #print(tokens['tokens'].shape)
        batch_size, num_words = tokens['tokens'].shape

        mask = get_text_field_mask(tokens)
        #print()
        doc = self._embed(tokens['tokens'])

        batch_size, num_words = mask.size()
        word_reps = doc.view(batch_size, num_words, -1)
        word_mask = mask.view(batch_size, num_words)

        sent_reps = self._word_cnn(word_reps, word_mask).view(batch_size, -1)

        sent_reps = self.dropout(sent_reps)
        clf = self._doc_project(sent_reps)

        output['prediction'] = [self._vocab.get_token_from_index(i.item(), 'labels') for i in clf.argmax(dim=-1)]
        output['loss'] = self._crit(clf, rating)
        self._acc(clf, rating)

        return output

    def get_metrics(self, reset=False):
        if not self.training:
            return {'accuracy': self._acc.get_metric(reset)}
        else:
            return {}

