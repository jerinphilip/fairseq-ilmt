from collections import defaultdict
from itertools import combinations, permutations
from . import DATASET_REGISTRY

def canonicalize(langcode):
    _variations = {
        "ur": ["ur", "ud"],
        "bn": ["bg", "bn"],
        "gu": ["gu", "gj"],
        "pa": ["pa", "pj"]
    }

    inverse = {}
    for root in _variations:
        for x in _variations[root]:
            inverse[x] = root

    return inverse.get(langcode, langcode)

def select(tags, splits, langs, direction):
    """
    """
    # Filter by split, langs
    registry = dict([
            (k, v)  \
            for k, v in DATASET_REGISTRY.items() \
            if k in tags
    ])

    # print(DATASET_REGISTRY)
    filtered_corpora = []

    for key in registry:
        _splits, f = registry[key]
        isplits = set(_splits).intersection(set(splits))
        isplits = list(isplits)
        for _split in isplits:
            #print(_split)
            corpora = f(_split)
            corpora = [
                c for c in corpora \
                if c.lang in langs
            ]

            filtered_corpora.extend(corpora)


    def group_by_tag(corpora):
        _dict = defaultdict(list)
        for corpus in corpora:
            _dict[corpus.tag].append(corpus)
        return _dict
    #print(filtered_corpora)
    corpora = group_by_tag(filtered_corpora)
    pairs = []
    req_pairs = []
    #print(corpora)
    for key in corpora:
        # TODO(jerin): Sort for determinism
        # TODO(shashanks): Check soundness
        for dx, dy in permutations(corpora[key], 2):
            src_lang, tgt_lang = dx[2], dy[2]
            if tgt_lang == direction['tgt']:
                req_pairs.append((dx, dy))             
            pairs.append((dx, dy))

    if direction['tgt']=='xx':
        return pairs

    else:
        return req_pairs


def pairs_select(corpora_config, split, direction):
    ls = []
    if split == 'valid': split = 'dev'
    for tag, v in corpora_config.items():
        tags = [tag]
        if split in v['splits']:
            splits = [split]
            pairs = select(tags, splits, v['langs'], direction)
            ls.extend(pairs)

    # Set is non-determinism. Sort
    def sort_key(pair):
        first, second = pair
        return (first.path, second.path)


    unique = list(set(ls))
    unique = sorted(unique, key=sort_key)
    return unique



if __name__ == '__main__':
    tags = ['iitb-hi-en', 'wat-ilmpc']
    splits = ['train']
    langs = ['en', 'hi', 'ta', 'ml']
    pairs = select(tags, splits, langs)
    from pprint import pprint
    pprint(pairs)

