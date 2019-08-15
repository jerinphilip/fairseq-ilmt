import os
from collections import namedtuple

ENV_VAR='ILMULTI_CORPUS_ROOT'
DATA_ROOT = os.environ.get(ENV_VAR, None)
if DATA_ROOT is None:
    raise Exception((
        "Please define {} in environment variable"
        .format(ENV_VAR)
        ))

DATASET_REGISTRY = {}
def dataset_register(tag, splits):
    def __inner(f):
        DATASET_REGISTRY[tag] = (splits, f)
        return f
    return __inner

def data_abspath(sub_path):
    path = os.path.join(DATA_ROOT, sub_path)
    return path

Corpus = namedtuple('Corpus', 'tag path lang')
def sanity_check(collection):
    for corpus in collection:
        print(corpus)

from . import corpora

