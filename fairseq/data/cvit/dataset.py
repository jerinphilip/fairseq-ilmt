from fairseq.data.indexed_dataset import IndexedRawTextDataset
import numpy as np
from copy import deepcopy
from .lmdb import LMDBCorpus, LMDBCorpusWriter
from multiprocessing import Pool

def language_token(lang):
    return '__t2{}__'.format(lang)

_flyweight = {}

class _CVITIndexedRawTextDataset:
    def __init__(self, corpus, tokenizer):
        self.corpus = corpus
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.read_data(corpus, tokenizer)
        self.size = len(self.tokens_list)

    def read_data(self, corpus, tokenizer):
        with open(corpus.path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                self.lines.append(line)
                _lang, tokens = tokenizer(line, lang=corpus.lang)
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens)+1)
        self.sizes = np.array(self.sizes, dtype=np.int32)
        print("Loaded {}".format(corpus.path))

    def __getitem__(self, idx):
        tokens = self.tokens_list[idx]
        return deepcopy(tokens)

    def __len__(self):
        return self.size

class CVITIndexedRawTextDataset(IndexedRawTextDataset):
    def __init__(self, corpus, tokenizer, dictionary, tgt_lang=None, append_eos=True, reverse_order=False):
        self.dictionary = dictionary
        self.tgt_lang = tgt_lang
        self.dataset = self._maybe_read(corpus, tokenizer)
        self.reverse_order = reverse_order
        self.append_eos = append_eos
        self.corpus = corpus
        
    @property   
    def corpus_id(self):
        #assuming corpus_name.split.lang
        lang = self.corpus.path.split('.')[-1]
        corpus_id = '{} {}'.format(self.corpus.tag,lang)
        return corpus_id

    def _maybe_read(self, corpus, tokenizer):
        path = corpus.path
        if path not in _flyweight:
            # Build LMDB corpus if not exists
            if not LMDBCorpus.exists(corpus):
                print("LMDB({}) does not exist. Building".format(corpus.path))
                raw_dataset = _CVITIndexedRawTextDataset(corpus, tokenizer)
                # writer = BufferedLMDBCorpusWriter(corpus, tokenizer, num_workers=30, max_size=1024*1024)
                writer = LMDBCorpusWriter(raw_dataset)
                writer.close()
                print("Built LMDB({})".format(corpus.path))

            _flyweight[path] = LMDBCorpus(corpus)

            # def debug_equivalence(raw_dataset, lmdb_dataset):
            #     assert((lmdb_dataset.sizes == raw_dataset.sizes).all())
            #     for i in range(len(raw_dataset)):
            #         assert(raw_dataset[i] == lmdb_dataset[i])

            # raw_dataset = _CVITIndexedRawTextDataset(corpus, tokenizer)
            # _flyweight[path] = raw_dataset
            # debug_equivalence(raw_dataset, _flyweight[path])
        return _flyweight[path]

    def __getitem__(self, index):
        tokens = self.dataset[index]
        if self.tgt_lang is not None:
            tokens.insert(0, language_token(self.tgt_lang))

        line = ' '.join(tokens)
        tokens = self.dictionary.encode_line(
            line, add_if_not_exist=False,
            append_eos=self.append_eos, reverse_order=self.reverse_order,
        ).long()
        # TODO(jerin): Cloning tokens seems to resolve memory leak.
        # Nope, doesn't
        return tokens.detach()

    @property
    def sizes(self):
        return self.dataset.sizes

    @property
    def size(self):
        return self.dataset.size

    def __len__(self):
        return len(self.dataset)
