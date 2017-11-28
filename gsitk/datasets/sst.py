"""
Processing of the Stanford Sentiment TreeBank dataset.

URL:
https://nlp.stanford.edu/sentiment/
REF:
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts

Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank

Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)
"""
import sys
import os
import logging
import shutil
import pandas as pd
import pytreebank
from gsitk.datasets import utils
from gsitk.datasets.datasets import Dataset
from gsitk.preprocess import normalize

logger = logging.getLogger(__name__)


class Sst(Dataset):
    polarities = {0:-2, 1:-1, 2:0, 3:1, 4:2}
    
    def _say_progress(self, subset, count):
        '''
        Just give me some sense of progress.
        '''
        logger.info('At {} in {}'.format(count, subset))

    def convert_treebank(self, trees, fold, progress=10000):
        parsed = pd.DataFrame(columns=['polarity', 'text', 'fold'])
        c = 0
        for line in trees:
            lab = line.label
            text = line.to_lines()[0]
            text = pytreebank.utils.normalize_string(text)
            parsed.loc[c, :] = [self.polarities[lab], text, fold]
            c += 1
            if c % progress == 0:
                self._say_progress(fold,c)
        return parsed

    def normalize_data(self):
        raw_datapath = os.path.join(self.data_path,
                                     self.info['properties']['data_file'])
        trees_path = os.path.join(self.data_path, 'trainDevTestTrees_PTB') 
        if not os.path.isdir(trees_path):
            os.mkdir(trees_path)
        shutil.move(raw_datapath, trees_path)
        stanford_treebank = pytreebank.load_sst(self.data_path)
        train = self.convert_treebank(stanford_treebank['train'], 'train')
        dev = self.convert_treebank(stanford_treebank['dev'], 'dev')
        test = self.convert_treebank(stanford_treebank['test'], 'test')
        data = pd.concat([train,dev,test], ignore_index=True)       
        
        # Remove directory to avoid pytreebank library error 
        #shutil.rmtree(raw_datapath)
        
        # Tokenize and clean the test
        text_data = normalize.normalize_text(data)
        logger.info(data)
        data['text'] = text_data

        return data
