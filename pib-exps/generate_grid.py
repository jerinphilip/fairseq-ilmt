import sys
from tqdm import tqdm
from argparse import ArgumentParser
from collections import defaultdict
from itertools import permutations
from indicnlp.contrib.wat.evaluate import Evaluator
from argparse import Namespace
import pandas as pd
import os

out_dir = './output/iter0'
model = 'm0'

def parallel_write(out_dir, corpus, src_lang, tgt_lang, hyp, ref):
    out_hyp = open('{}/grid/{}/pib_m0.{}-{}.hyp'.format(out_dir, corpus, src_lang, tgt_lang),'a')
    out_ref = open('{}/grid/{}/pib_m0.{}-{}.ref'.format(out_dir, corpus, src_lang, tgt_lang),'a')
    print(hyp,file=out_hyp)
    print(ref,file=out_ref)


def generate_pairs(corpus, ind, hyp, ref):
    for ind, hyp, ref in tqdm(zip(ind, hyp, ref)):
        ind = ind.split('\n')[0]
        tag = ind.split()[0]
        direction = ind.split()[1]
        direction = direction.split('_') #xx_yy
        hyp_line = hyp.split('\n')[0]
        ref_line = ref.split('\n')[0]
        if corpus == tag:
            parallel_write(out_dir, corpus, direction[0], direction[1], hyp_line, ref_line)



def generate_grid(corpus):
    langs = ['en', 'hi', 'ta', 'te', 'ml', 'ur', 'bn', 'mr', 'gu', 'mr', 'pa', 'or']
    data = defaultdict(float)
    for lang in langs:
        data[lang] = 0.00

    df = pd.DataFrame(data, index = langs)
    perm = permutations(langs, 2) 
    for i in list(perm):
        args = Namespace(hypothesis='{}/grid/{}/pib_m0.{}-{}.hyp'.format(out_dir, corpus, i[0], i[1]) \
                       ,references=['{}/grid/{}/pib_m0.{}-{}.ref'.format(out_dir, corpus, i[0], i[1])])
        try:
            evaluator = Evaluator.build(args)
            stats = evaluator.run()
            for key, val in stats.items():
                df.at[i[0],i[1]] = float(val[7:11])
        except:
            df.at[i[0],i[1]] = 0
            pass
            
    df.to_csv('{}/grid/{}/{}_grid.csv'.format(out_dir, corpus, model))


corpora = ['iitb-hi-en',
           'ilci',
           'ufal-en-tam',
           'wat-ilmpc',
           'odiencorp',
           'bible-en-te'
        ]

if __name__ == '__main__':
    for corpus in corpora:
        path = '{}/grid/{}'.format(out_dir, corpus)
        try:
            os.mkdir(path)
        except OSError as error: 
            pass
        ind = open('{}/pib_m0.ind'.format(out_dir),'r')
        hyp = open('{}/pib_m0.hyp'.format(out_dir),'r')
        ref = open('{}/pib_m0.ref'.format(out_dir),'r')
        generate_pairs(corpus, ind, hyp, ref)
        generate_grid(corpus)