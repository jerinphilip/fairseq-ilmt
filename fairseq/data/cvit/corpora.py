from . import DATASET_REGISTRY
from . import dataset_register, data_abspath
from . import Corpus, sanity_check
from itertools import permutations, combinations


@dataset_register('iitb-hi-en', ['train', 'dev', 'test'])
def IITB_meta(split):
    corpora = []
    for lang in ['en', 'hi']:
        sub_path = 'filtered-iitb/{}.{}'.format(split, lang)
        corpus = Corpus('iitb-hi-en', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('national-newscrawl', ['train', 'dev', 'test'])
def NationalNewscrawl_meta(split):
    if split in ['dev', 'test']:
        return []
    corpora = []
    for lang in ['en', 'hi']:
        sub_path = 'national-newscrawl/national.{}'.format(lang)
        #corpus = Corpus('iitb-hi-en', data_abspath(sub_path), lang)
        corpus = Corpus('national-newscrawl', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('wat-ilmpc', ['train', 'dev', 'test'])
def WAT_meta(split):
    corpora = []
    langs = ['bn', 'hi', 'ml', 'ta', 'te', 'ur']
    for lang in langs:
        for src in [lang, 'en']:
            sub_path = 'indic_languages_corpus/bilingual/{}-en/{}.{}'.format(
                    lang, split, src
            )
            corpus_name = 'wat-ilmpc-{}-{}'.format(lang, 'en')
            corpus = Corpus(corpus_name, data_abspath(sub_path), src)
            corpora.append(corpus)
    return corpora

@dataset_register('pib-test', ['test'])
def PIBTEST_meta(split):
    corpora = []
    langs = ['hi', 'ta', 'te', 'ml', 'ur', 'bn', 'gu', 'mr', 'pa', 'or']
    for lang in langs:
        fst, snd = sorted([lang, 'en'])
        dirname = '{}-{}'.format(fst, snd)
        for src in [lang, 'en']:
                sub_path = 'pib-test/{}/{}.{}'.format(
                        dirname, split, src
                )
                corpus_name = 'pib-test-{}-{}'.format(lang, 'en')
                corpus = Corpus(corpus_name, data_abspath(sub_path), src)
                corpora.append(corpus)
    return corpora

@dataset_register('pib', ['train', 'dev', 'test'])
def PIB_meta(split):
    if split in ['dev', 'test']:
        return []

    corpora = []
    langs = ['hi', 'ta', 'te', 'ml', 'ur', 'bn', 'gu', 'mr', 'pa', 'or', 'en']
    langs = sorted(langs)
    perm = combinations(langs, 2)
    for src, tgt in list(perm):
        for lang in [src, tgt]:
                sub_path = 'pib/{}-{}/{}.{}'.format(
                        src, tgt, 'train', lang
                )
                corpus_name = 'pib-{}-{}'.format(src, tgt)
                corpus = Corpus(corpus_name, data_abspath(sub_path), lang)
                corpora.append(corpus)
    return corpora

@dataset_register('pib-v1', ['train', 'dev', 'test'])
def PIBV2_meta(split):
    if split in ['dev', 'test']:
        return []

    corpora = []
    langs = ['hi', 'ta', 'te', 'ml', 'ur', 'bn', 'gu', 'mr', 'pa', 'or', 'en']
    langs = sorted(langs)
    perm = combinations(langs, 2)
    for src, tgt in list(perm):
        for lang in [src, tgt]:
                sub_path = 'pib-v1/{}-{}/{}.{}'.format(
                        src, tgt, 'train', lang
                )
                corpus_name = 'pib-v1-{}-{}'.format(src, tgt)
                corpus = Corpus(corpus_name, data_abspath(sub_path), lang)
                corpora.append(corpus)
    return corpora

@dataset_register('pib-v2', ['train', 'dev', 'test'])
def PIBV2_meta(split):
    if split in ['dev', 'test']:
        return []

    corpora = []
    langs = ['hi', 'ta', 'te', 'ml', 'ur', 'bn', 'gu', 'mr', 'pa', 'or', 'en']
    langs = sorted(langs)
    perm = combinations(langs, 2)
    for src, tgt in list(perm):
        for lang in [src, tgt]:
                sub_path = 'pib-v2/{}-{}/{}.{}'.format(
                        src, tgt, 'train', lang
                )
                corpus_name = 'pib-v2-{}-{}'.format(src, tgt)
                corpus = Corpus(corpus_name, data_abspath(sub_path), lang)
                corpora.append(corpus)
    return corpora

@dataset_register('mkb', ['train', 'dev', 'test'])
def MKB_meta(split):
    if split in ['train', 'dev']:
        return []

    corpora = []
    langs = ['ml', 'ur', 'te', 'hi', 'ta', 'bn', 'gu', 'or', 'mr', 'en']
    langs = sorted(langs)
    perm = combinations(langs, 2)
    for src, tgt in list(perm):
        for lang in [src, tgt]:
                sub_path = 'mkb/{}-{}/{}.{}'.format(
                        src, tgt, 'mkb', lang
                )
                corpus_name = 'mkb-{}-{}'.format(src, tgt)
                corpus = Corpus(corpus_name, data_abspath(sub_path), lang)
                corpora.append(corpus)
    return corpora

@dataset_register('ufal-en-tam', ['train', 'dev', 'test'])
def UFALEnTam_meta(split):
    corpora = []
    for lang in ['en', 'ta']:
        sub_path = 'ufal-en-tam/{}.{}.{}'.format('corpus.bcn', split, lang)
        corpus = Corpus('ufal-en-tam', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora


@dataset_register('ilci', ['train', 'dev', 'test'])
def ILCI_meta(split):
    #if split in ['dev', 'test']:
    #    return []
    corpora = []
    langs = [
        'en', 'te', 'hi', 'ml', 
        'ta', 'ud', 'bg', 'mr',
        'gj', 'pj', 'kn'
    ]

    from .utils import canonicalize

    for lang in langs:
        sub_path = 'ilci/{}.{}'.format(split, lang)
        _lang = canonicalize(lang)
        corpus = Corpus('ilci', data_abspath(sub_path), _lang)
        corpora.append(corpus)
    return corpora

@dataset_register('bible-en-te', ['train', 'dev', 'test'])
def BIBLEEnTe_meta(split):
    corpora = []
    for lang in ['en', 'te']:
        sub_path = 'bible-en-te/{}.{}.{}'.format('bible', split, lang)
        corpus = Corpus('bible-en-te', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora


@dataset_register('eenadu-en-te', ['train'])
def EenaduBacktrans_meta(split):
    if split in ['dev', 'test']:
        return []

    corpora = []
    for lang in ['en','te']:
        sub_path = 'eenadu-en-te/train.{}'.format(lang)
        corpus = Corpus('eenadu-en-te', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('newstest2019guen', ['test'])
def Newstest2019guen_meta(split):
    if split in ['train', 'dev']:
        return []

    corpora = []
    for lang in ['en','gu']:
        sub_path = 'newstest2019guen/test.{}'.format(lang)
        corpus = Corpus('newstest2019guen', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('newstest2019engu', ['test'])
def Newstest2019engu_meta(split):
    if split in ['train', 'dev']:
        return []

    corpora = []
    for lang in ['en','gu']:
        sub_path = 'newstest2019engu/test.{}'.format(lang)
        corpus = Corpus('newstest2019engu', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

@dataset_register('odiencorp', ['train', 'dev', 'test'])
def OdiEnCorp_meta(split):
    corpora = []
    for lang in ['en', 'or']:
        sub_path = 'odiencorp/{}.{}'.format(split, lang)
        corpus = Corpus('odiencorp', data_abspath(sub_path), lang)
        corpora.append(corpus)
    return corpora

if __name__ == '__main__':
    def merge(*_as):
        _ase = []
        for a in _as:
            _ase.extend(a)
        return _ase

    ls = []
    for key in DATASET_REGISTRY:
        splits, f = DATASET_REGISTRY[key]
        for split in splits:
            ls.append(f(split))

    _all = merge(*ls)
    sanity_check(_all)

