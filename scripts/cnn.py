import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, batch_size,
                 output_size,
                 num_filters,
                 kernels,
                 keep_probab,
                 vocab_size,
                 embedding_length):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.num_filters = num_filters
        self.kernels = kernels
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.convs = []
        for kernel in kernels:
            self.convs.append(nn.Conv2d(1,
                                        num_filters,
                            (kernel, embedding_length)))
        self.dropout = nn.Dropout(keep_probab)
        self.label = nn.Linear(len(kernels)*num_filters, output_size)
        
    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)
        
        return max_out
    
    def forward(self, input_sentences, batch_size=None):
        input = self.word_embeddings(input_sentences)
        # input.size() = (batch_size, num_seq, embedding_length)
        input = input.unsqueeze(1)
        # input.size() = (batch_size, 1, num_seq, embedding_length)
        max_out = []
        for conv in self.convs:
            max_out.append(self.conv_block(input, conv))
        all_out = torch.cat(max_out, 1)
        # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)
        # fc_in.size()) = (batch_size, num_kernels*out_channels)
        logits = self.label(fc_in)
        
        return logits
