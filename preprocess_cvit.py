from argparse import ArgumentParser
from fairseq.data.cvit.utils import pairs_select
from fairseq.data.cvit.dataset import _CVITIndexedRawTextDataset
from fairseq.data.cvit.lmdb import LMDBCorpusWriter, LMDBCorpus
import yaml
from multiprocessing import Pool
from functools import partial
import os
from pprint import pprint
from ilmulti.sentencepiece import build_tokenizer

def read_config(path):
    with open(path) as config:
        contents = config.read()
        data = yaml.load(contents)
        return data

def build_corpus(corpus):
	if not LMDBCorpus.exists(corpus):
		print("LMDB({}) does not exist. Building".format(corpus.path))
		raw_dataset = _CVITIndexedRawTextDataset(corpus, tokenizer)
		# writer = BufferedLMDBCorpusWriter(corpus, tokenizer, num_workers=30, max_size=1024*1024)
		writer = LMDBCorpusWriter(raw_dataset)
		writer.close()
		print("Built LMDB({})".format(corpus.path))


def get_pairs(data, splits, direction):
	corpora = []
	for split in splits:
		pairs = pairs_select(data['corpora'], split, direction)
		srcs,tgts = list(zip(*pairs))
		corpora.extend(srcs)
		corpora.extend(tgts)
	
	return list(set(corpora))

def mp(build_corpus , corpora):
	pool = Pool(processes=os.cpu_count())
	pool.map_async(build_corpus , corpora)
	pool.close()
	pool.join()

if __name__ == '__main__':
	parser=ArgumentParser()
	parser.add_argument('data')
	args = parser.parse_args()
	splits = []
	data = read_config(args.data)
	for corpus in data['corpora']:
		splits.extend(data['corpora'][corpus]['splits'])
	direction = data['direction']
	tokenizer_tag = data['tokenizer']
	splits = list(set(splits))
	corpora = get_pairs(data, splits, direction)
	tokenizer = build_tokenizer(tokenizer_tag)
	mp(build_corpus , corpora)



      
