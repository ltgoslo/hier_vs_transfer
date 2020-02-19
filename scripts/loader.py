import os
import tarfile

from overrides import overrides

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.fields import TextField, LabelField, ListField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token

from allennlp.data.dataset_readers import DatasetReader


@DatasetReader.register('norec-hier')
class NorecReaderHierarchical(DatasetReader):
    def __init__(self, num_examples: int = -1) -> None:
        super().__init__()
        self._tokenizer = JustSpacesWordSplitter()
        self._indexer = {'tokens': SingleIdTokenIndexer('tokens')}
        self._num_examples = num_examples

    @overrides
    def _read(self, file_dir: str):
        # file_dir should point to the conllu.tar.gz file plus train, dev, or test
        # example: file_dir="data/en/conllu.tar.gz/train"
        file, split = os.path.split(file_dir)

        tar = tarfile.open(file, "r:gz")
        file_names = [tarinfo for tarinfo in tar.getmembers() if split in tarinfo.name and ".conllu" in tarinfo.name]

        if split == "train" and self._num_examples > -1:
            file_names = file_names[:self._num_examples]


        for fname in file_names:
            content = tar.extractfile(fname)
            language = content.readline().decode("utf8").rstrip("\n")[-2:]
            rating = content.readline().decode("utf8").rstrip("\n")[-1]
            doc_id = content.readline().decode("utf8").rstrip("\n").split()[-1]

            doc, words = [], []
            for line in content:
                line = line.decode("utf8")
                if line[0] == '#':
                    continue

                if not line.rstrip("\n"):
                    doc.append(words)
                    words = []
                    continue

                else:
                    words.append(Token(line.split("\t")[1]))

            # reload idk
            #content = tar.extractfile(fname).read()
            yield self.text_to_instance(doc, doc_id, rating)

    @overrides
    def text_to_instance(self, doc, doc_id, rating) -> Instance:
        fields = {}

        fields['rating'] = LabelField(rating)
        fields['doc'] = ListField([TextField(sent, self._indexer) for sent in doc])

        nsents = len(doc)
        ntokens = sum(len(i) for i in doc)
        fields['meta'] = MetadataField({'doc_id': doc_id,
                                        'sentences': nsents,
                                        'tokens': ntokens})
        return Instance(fields)


@DatasetReader.register('norec-flat')
class NorecReader_Flat(DatasetReader):
    def __init__(self, num_examples: int = -1) -> None:
        super().__init__()
        self._tokenizer = JustSpacesWordSplitter()
        self._indexer = {'tokens': SingleIdTokenIndexer('tokens')}
        self._num_examples = num_examples

    @overrides
    def _read(self, file_dir: str):
        # file_dir should point to the conllu.tar.gz file plus train, dev, or test
        # example: file_dir="data/en/conllu.tar.gz/train"
        file, split = os.path.split(file_dir)

        tar = tarfile.open(file, "r:gz")
        file_names = [tarinfo for tarinfo in tar.getmembers() if split in tarinfo.name and ".conllu" in tarinfo.name]

        if split == "train" and self._num_examples > -1:
            file_names = file_names[:self._num_examples]


        for fname in file_names:
            content = tar.extractfile(fname)
            language = content.readline().decode("utf8").rstrip("\n")[-2:]
            rating = content.readline().decode("utf8").rstrip("\n")[-1]
            doc_id = content.readline().decode("utf8").rstrip("\n").split()[-1]

            tokens = []
            num_sents = 0
            num_tokens = 0

            for line in content:
                line = line.decode("utf8")
                if line[0] == '#':
                    continue

                if not line.rstrip("\n"):
                    num_sents += 1
                    continue

                else:
                    tokens.append(Token(line.split("\t")[1]))
                    num_tokens += 1

            #content = tar.extractfile(fname).read()
            yield self.text_to_instance(tokens, doc_id, rating, num_sents, num_tokens)

    @overrides
    def text_to_instance(self, tokens, doc_id, rating, num_sents, num_tokens) -> Instance:
        fields = {}

        fields['rating'] = LabelField(rating)
        fields['tokens'] = TextField(tokens, self._indexer)
        fields['meta'] = MetadataField({'tokens': num_tokens,
                                        'doc_id': doc_id,
                                        'sentences': num_sents})
        return Instance(fields)
