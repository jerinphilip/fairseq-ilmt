import os
import lmdb
import numpy as np
import torch
from copy import deepcopy
from multiprocessing import Pool
import sys

class LMDBCorpusWriter:
    def __init__(self, raw_dataset):
        self.corpus = raw_dataset.corpus
        path = '{}.processed.lmdb'.format(self.corpus.path)
        map_size = LMDBCorpusWriter.corpus_map_size(self.corpus)
        self.env = lmdb.open(path, map_size=map_size)
        self._write_corpus(raw_dataset)

    def _write_corpus(self, raw_dataset):
        self.write_metadata(raw_dataset.sizes, len(raw_dataset))
        self._write_samples(raw_dataset)

    def _write_samples(self, raw_dataset):
        with self.env.begin(write=True) as txn:
            for idx, sample in enumerate(raw_dataset.tokens_list):
                key = '{}'.format(idx)
                encoded_sample = ' '.join(sample).encode()
                key = key.encode("ascii")
                txn.put(key, encoded_sample) 

    @staticmethod
    def corpus_map_size(corpus):
        scale = 20
        size = int(scale*os.path.getsize(corpus.path))
        return size

    def _set(self, key, val):
        with self.env.begin(write=True) as txn:
            key = key.encode("ascii")
            txn.put(key, val)
    
    def write_metadata(self, sizes, num_samples):
        num_samples = '{}'.format(num_samples).encode("ascii")
        self._set("num_samples", num_samples)
        self._set("sizes", sizes.tobytes())

    def close(self):
        self.env.close()


class LMDBCorpus:
    def __init__(self, corpus):
        self.corpus = corpus
        map_size = LMDBCorpusWriter.corpus_map_size(corpus)
        path = '{}.processed.lmdb'.format(corpus.path)
        self.env = lmdb.open(path, map_size=map_size, readonly=True)
        self._init_metadata()
        print("Initialized LMDB: {}".format(corpus.path))

    def _init_metadata(self):
        sizes = self._get_value("sizes")
        num_samples = self._get_value("num_samples")
        self.sizes = np.frombuffer(sizes, dtype=np.int32)
        self.num_samples = int(num_samples.decode("ascii"))


    def _get_value(self, key):
        key = key.encode("ascii")
        with self.env.begin() as txn:
            record = txn.get(key)
            return record
        return None

    def __len__(self):
        return self.num_samples

    @property   
    def corpus_id(self):
        corpus_id = self.corpus.path.split('.')
        #corpus_id = self.corpus.path.replace('/','_')
        return corpus_id[1]

    def __getitem__(self, idx):
        _key = '{}'.format(idx)
        record = self._get_value(_key)
        
        cached = deepcopy(record.decode('utf-8'))
        tokens = cached.split()
        return tokens

    @staticmethod
    def exists(corpus):
        path = '{}.processed.lmdb'.format(corpus.path)
        return os.path.exists(path)




