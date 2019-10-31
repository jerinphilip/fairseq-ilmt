import sys
from tqdm import tqdm
from argparse import ArgumentParser
import yaml
from collections import defaultdict
from itertools import permutations
from fairseq.data.cvit.utils import pairs_select, select
from fairseq.data.cvit.dataset import _CVITIndexedRawTextDataset
from fairseq.data.cvit.lmdb import LMDBCorpusWriter, LMDBCorpus
#from indicnlp.contrib.wat.evaluate import Evaluator
from wsacrebleu.evaluate import Evaluator
from argparse import Namespace
import pandas as pd
import os
from pprint import pprint

class ParallelWriter:
    def __init__(self, fpath, fname):
        self.fpath = fpath
        self.fname = fname
        self.files = {}

    def get_fp(self, src, tgt):
        
        if not os.path.exists(self.fpath):
            os.makedirs(self.fpath)

        if (src, tgt) in self.files:
            return self.files[(src, tgt)]

        self.files[(src, tgt)] = [
            open(os.path.join(self.fpath, '{}.hyp'.format(self.fname)), 'w'),
            open(os.path.join(self.fpath, '{}.ref'.format(self.fname)), 'w')
        ]
        return self.files[(src, tgt)]

    def write(self, src, tgt, srcline, tgtline):
        srcfile, tgtfile = self.get_fp(src, tgt)
        print(srcline, file=srcfile)
        print(tgtline, file=tgtfile)

def read_config(path):
    with open(path) as config:
        contents = config.read()
        data = yaml.load(contents)
        return data

def generate_pairs(ind, hyp, ref, out_dir):
    def canonicalize_corpus_tag(corpus):
        corpus_tag = {'wat-ilmpc':'wat-ilmpc', 'mkb':'mkb', 'pib-test':'pib-test'}
        for name in corpus_tag:
            if name in corpus:
                return corpus_tag[name]
        return corpus
    
    export = defaultdict(lambda: defaultdict(list))
    for ind, hyp, ref in tqdm(zip(ind, hyp, ref)):           
        ind = ind.rstrip()
        corpus_tag, direction = ind.split()
        hyp_line = hyp.rstrip()
        ref_line = ref.rstrip()
        corpus_tag = canonicalize_corpus_tag(corpus_tag)
        export[corpus_tag][direction].append([hyp_line, ref_line])   
    
    for corpus_tag in export:
        fpath = os.path.join(out_dir, corpus_tag)
        for direction in export[corpus_tag]:
            src_lang, tgt_lang = direction.split('_') #xx_yy
            fname = '{}-{}'.format(src_lang, tgt_lang)
            pwriter = ParallelWriter(fpath, fname)
            for line in export[corpus_tag][direction]:
                hyp_line, ref_line = line
                '''
                # Only in case of MKB test on iter0 
                if corpus_tag=='mkb' and tgt_lang=='ur' and src_lang!='en':
                    hyp = hyp_line.split()
                    hyp.reverse()
                    hyp_line = " ".join(hyp)
                '''
                pwriter.write(src_lang, tgt_lang, hyp_line, ref_line)

def generate_grid(corpus, pairs, langs, direction, out_dir):
    data = defaultdict(float)
    langs = sorted(langs)
    df = pd.DataFrame(data, index=langs)
    perm = permutations(langs, 2)
    hyp_path = os.path.join(out_dir, corpus)

    def reference_path(pairs, src_lang, tgt_lang):
        for pair in pairs:
            src, tgt = pair
            slang, tlang = src[2], tgt[2]
            ref_path = tgt[1]
            if src_lang==slang and tgt_lang==tlang:
                return ref_path

    for (src_lang, tgt_lang) in list(perm):
        reference = reference_path(pairs, src_lang, tgt_lang)

        args = Namespace(hypothesis='{}/{}-{}.hyp'.format(hyp_path, src_lang, tgt_lang) \
                       ,references=[reference], lang=tgt_lang)
        evaluator = Evaluator.build(args)
        stats = evaluator.run()
        for key, val in stats.items():
            print(corpus, src_lang, tgt_lang, val)
            df.at[src_lang, tgt_lang] = float(val[7:12])

    df = df.sort_index(axis=1)                    
    df.to_csv('{}/grid.csv'.format(hyp_path))

if __name__ == '__main__':
    parser=ArgumentParser()
    parser.add_argument('--ind', help='index file', required=True)
    parser.add_argument('--hyp', help='hypothesis', required=True)
    parser.add_argument('--ref', help='reference', required=True)
    parser.add_argument('--out_dir', help='output dir', required=True)
    parser.add_argument('--test_config', help = 'config file used for test', required=True)
    args = parser.parse_args()
    ind = open(args.ind,'r')
    hyp = open(args.hyp,'r') 
    ref = open(args.ref,'r')
    out_dir = args.out_dir
    
    generate_pairs(ind, hyp, ref, out_dir)

    data = read_config(args.test_config)
    splits = ['test']
    direction = data['direction']
    
    for corpus in data['corpora']:
        langs = data['corpora'][corpus]['langs']        
        pairs = select(corpus, splits, langs, direction)
        generate_grid(corpus, pairs, langs, direction, out_dir)

    
    