# Comparing hierarchical and transfer learning-based approaches for document-level sentiment classification

Jeremy Barnes [jeremycb@ifi.uio.no]

State-of-the-art models to document-level sentiment analysis fall into one of two approaches: 1) transfer learning approaches, such as ULMFit, BERT, etc, which first pretrain a language model on large amounts of unlabeled data and then finetune on the task, and 2) hierarchical approaches which instead attempt to model the relative contributions of each sentence to the overall polarity of a document.


## Models
1. BOW model
2. CNN
3. Hierarchical CNN
4. [ULMFiT](https://aclweb.org/anthology/P18-1031)
5. [Hierarchical Attention Network](https://www.aclweb.org/anthology/N16-1174)

## Languages

1. Norwegian
2. English
3. German
4. Japanese
5. French

### Requirements

1. allennlp
2. sklearn  ```pip install -U scikit-learn```
3. Pytorch ```pip install torch torchvision```
4. nltk ```pip install nltk```


