import sys
from tqdm import tqdm
from argparse import ArgumentParser
import yaml
from collections import defaultdict
from itertools import permutations
from indicnlp.contrib.wat.evaluate import Evaluator
from argparse import Namespace
import pandas as pd
import os

model = 'm0'

def read_config(path):
    with open(path) as config:
        contents = config.read()
        data = yaml.load(contents)
        return data

def get_langs(data):
    corpus_list = defaultdict(list)
    for corpus in data['corpora']:
        corpus_list[corpus] = data['corpora'][corpus]['langs']
    return corpus_list

def parallel_write(corpus, src_lang, tgt_lang, hyp, ref, out_dir):
    out_hyp = open('{}/{}/pib_m0.{}-{}.hyp'.format(out_dir, corpus, src_lang, tgt_lang),'a')
    out_ref = open('{}/{}/pib_m0.{}-{}.ref'.format(out_dir, corpus, src_lang, tgt_lang),'a')
    print(hyp,file=out_hyp)
    print(ref,file=out_ref)



def generate_pairs(ind, hyp, ref, out_dir):
    for ind, hyp, ref in tqdm(zip(ind, hyp, ref)):           
        ind = ind.split('\n')[0]
        corpus_tag = ind.split()[0]
        direction = ind.split()[1]
        direction = direction.split('_') #xx_yy
        src_lang, tgt_lang = direction[0], direction[1]
        hyp_line = hyp.split('\n')[0]
        ref_line = ref.split('\n')[0]
        if 'wat-ilmpc' not in corpus_tag:
            parallel_write(corpus_tag, src_lang, tgt_lang, hyp_line, ref_line, out_dir)
        else:
            corpus_tag = 'wat-ilmpc'
            parallel_write(corpus_tag, src_lang, tgt_lang, hyp_line, ref_line, out_dir)


def generate_grid(out_dir, corpus):
    langs = ['en', 'hi', 'ta', 'te', 'ml', 'ur', 'bn', 'mr', 'gu', 'mr', 'pa', 'or']
    data = defaultdict(float)
    for lang in langs:
        data[lang] = 0.00

    df = pd.DataFrame(data, index = langs)
    perm = permutations(langs, 2) 
    for i in list(perm):
        args = Namespace(hypothesis='{}/{}/pib_m0.{}-{}.hyp'.format(out_dir, corpus, i[0], i[1]) \
                       ,references=['{}/{}/pib_m0.{}-{}.ref'.format(out_dir, corpus, i[0], i[1])])
        try:
            evaluator = Evaluator.build(args)
            stats = evaluator.run()
            for key, val in stats.items():
                df.at[i[0],i[1]] = float(val[7:11])
        except:
            df.at[i[0],i[1]] = 0
            pass            
    df.to_csv('{}/grid/{}/{}_grid.csv'.format(out_dir, corpus, model))


def create_dir(out_dir, corpus):
    path = '{}/{}'.format(out_dir, corpus)
    try:
        os.makedirs(path)
    except OSError as error: 
        pass

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
    data = read_config(args.test_config)
    corpora = get_langs(data)
    for corpus in corpora:
        create_dir(out_dir, corpus)

    generate_pairs(ind, hyp, ref, out_dir)
    #for corpus in corpora:
    #    generate_grid(out_dir, corpus)

    
    