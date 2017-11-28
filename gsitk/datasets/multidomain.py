"""
Processing of the  Multi-Domain Sentiment Dataset (version 2.0) dataset.

URL:
https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
REF:
John Blitzer, Mark Dredze, Fernando Pereira. Biographies, Bollywood, Boom-boxes and Blenders

Domain Adaptation for Sentiment Classification

Association of Computational Linguistics (ACL), 2007.
"""

import os
import logging
import codecs
import pandas as pd
from bs4 import BeautifulSoup
from glob import glob
from gsitk.datasets.datasets import Dataset
from gsitk.preprocess import simple

logger = logging.getLogger(__name__)


class Multidomain(Dataset):
    polarities = {
        5.0: 2,
        4.0: 1,
        3.0: 0,
        2.0: -1,
        1.0: -2
    }

    def read_file(self, file_path):
        texts = []
        ratings = []

        if not os.path.exists(file_path):
            return [], []

        with codecs.open(file_path, 'r', encoding='iso-8859-2') as f:
            parser = BeautifulSoup(f, 'lxml')
            
        for text in parser.find_all('review_text'):
            text = simple.clean_str(text.text).split(' ')
            texts.append(text)
            
        for rating in parser.find_all('rating'):
            rating = float(rating.text.strip())
            ratings.append(rating)
            
        return texts, ratings

    def read_category(self, category_path, category):  
        df = pd.DataFrame(columns=['polarity', 'text', 'category'])
    
        pol_conv = {
            5.0: 1,
            4.0: 1,
            3.0: 0,
            2.0: -1,
            1.0: -1
        }
    
        positive_name, negative_name = 'positive.review', 'negative.review'
        positive_texts, positive_ratings = self.read_file(os.path.join(category_path,
                                                                       positive_name))
        negative_texts, negative_ratings = self.read_file(os.path.join(category_path,
                                                                       negative_name))
        
        texts = positive_texts + negative_texts
        ratings = positive_ratings + negative_ratings
        assert len(texts) == len(ratings)
        df['polarity'] = ratings
        df['polarity'] = df['polarity'].apply(lambda p: pol_conv[p])
        df['text'] = texts
        df['category'] = [category] * len(ratings)
        
        return df

    def normalize_data(self):
        categories_path = os.path.join(self.data_path, self.info['properties']['data_file'])

        categories_df = []
        for category_path in glob(os.path.join(categories_path, '*')):
            if not os.path.isdir(category_path):
                continue
            category_name = os.path.basename(category_path)
            categories_df.append(self.read_category(category_path, category_name))
        data = pd.concat(categories_df, ignore_index=True)
        return data



