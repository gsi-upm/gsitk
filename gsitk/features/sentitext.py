from nltk.corpus import stopwords
from gsitk.features.sentiwn import SentiWordNet
import re
import nltk
import csv
import sys
import unicodedata
import tarfile
import logging
import string
import pandas as pd
import numpy as np
import nltk
from nltk import bigrams
from nltk import trigrams
from nltk.tokenize import word_tokenize
import os
from six.moves import urllib
from nltk.corpus import wordnet as wn
from nltk import word_tokenize
from gsitk.preprocess.pprocess_twitter import tokenize
from gsitk.config import default

config = default()

logger = logging.getLogger(__name__)


def delete_special_chars(text):
    regHttp = re.compile('(http://)[a-zA-Z0-9]*.[a-zA-Z0-9/]*(.[a-zA-Z0-9]*)?')
    regHttps = re.compile('(https://)[a-zA-Z0-9]*.[a-zA-Z0-9/]*(.[a-zA-Z0-9]*)?')
    regAt = re.compile('@([a-zA-Z0-9]*[*_/&%#@$]*)*[a-zA-Z0-9]*')
    text = re.sub(regHttp, '', text)
    text = re.sub(regAt, '', text)
    text = re.sub('RT : ', '', text)
    text = re.sub(regHttps, '', text)
    text = re.sub('[0-9]', '', text)
    return text.strip()

def load_stopwords(folder=config.RESOURCES_PATH):
    stopwords = list()
    with open(os.path.join(folder, 'sentitext_stopwords.txt'),"rt") as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(re.sub("\n","",line))
    return stopwords

def extract_ngrams(texts):
    pos_dictionary = {'JJR':'JJ','JJS':'JJ','NNS':'NN','NNP':'NN','NNPS':'NN',
              'PRP$':'PRP','RBD':'RB','RBS':'RB','RBP':'RB','VBZ':'VB',
              'VBD':'VB','VBP':'VB','VBN':'VB','VBG':'VB'}
    stopwords = load_stopwords()
    wnl = nltk.WordNetLemmatizer()
    unigrams_words = []
    unigrams_lemmas = []
    pos_tagged = []
    for text in texts:
        text = delete_special_chars(text)
        tokens = word_tokenize(text)
        words_pos = nltk.pos_tag(tokens)
        for word_pos in words_pos:
            if word_pos[0] not in stopwords or word_pos[0] not in string.punctuation:
                unigrams_words.append(word_pos[0].lower())
                unigrams_lemmas.append(wnl.lemmatize(word_pos[0]).lower())
                if word_pos[1] not in pos_dictionary:
                    pos_tagged.append(word_pos[1])
                else:
                    pos_tagged.append(pos_dictionary[word_pos[1]])
    return unigrams_words,unigrams_lemmas,pos_tagged

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def extract_feature_tokens(tokens,bag_of_tokens):
    feature_tokens = {}
    tokens_ngrams=[]
    auxiliar_tokens=[]
    sent_punctuation = [".",",", ";", "!", "?", ":", "\n", "\r"]
    negations = ["no", "neither", "never", "nothing",
                "anyone", "any","fail", "lack", "without","dont","don","t"]
    negation=False
    for token in tokens:
        if token in negations:
            tokens_ngrams += [gram for gram in auxiliar_tokens if gram not in stopwords]
            tokens_ngrams += ['_'.join(bigram) for bigram in find_ngrams(auxiliar_tokens,2)]
            tokens_ngrams += ['_'.join(trigram) for trigram in find_ngrams(auxiliar_tokens,3)]
            if len(tokens_ngrams)>0 and negation:
                for gram in tokens_ngrams:
                    if gram in bag_of_tokens and gram not in feature_tokens:
                        feature_tokens['has_NEG_'+gram] = True
            elif len(tokens_ngrams)>0:
                for gram in tokens_ngrams:
                    if gram in bag_of_tokens and gram not in feature_tokens:
                        feature_tokens['has_'+gram] = True
            if negation:
                negation=False
            else:
                negation=True
            tokens_ngrams=[]
            auxiliar_tokens=[]
            auxiliar_tokens.append(token)
        elif token in sent_punctuation:
            if negation:
                tokens_ngrams += [gram for gram in auxiliar_tokens if gram not in stopwords]
                tokens_ngrams += ['_'.join(bigram) for bigram in find_ngrams(auxiliar_tokens,2)]
                tokens_ngrams += ['_'.join(trigram) for trigram in find_ngrams(auxiliar_tokens,3)]
                if len(tokens_ngrams)>0:
                    for gram in tokens_ngrams:
                        if gram in bag_of_tokens and gram not in feature_tokens:
                            feature_tokens['has_NEG_'+gram] = True
                negation=False
                tokens_ngrams=[]
                auxiliar_tokens=[]
            else:
                auxiliar_tokens.append(token)
        elif len(token) > 0:
            auxiliar_tokens.append(token)
            
    if len(auxiliar_tokens) > 0:
        tokens_ngrams += [gram for gram in auxiliar_tokens if gram not in stopwords]
        tokens_ngrams += ['_'.join(bigram) for bigram in find_ngrams(auxiliar_tokens,2)]
        tokens_ngrams += ['_'.join(trigram) for trigram in find_ngrams(auxiliar_tokens,3)]
        if len(tokens_ngrams)>0 and negation:
            for gram in tokens_ngrams:
                if gram in bag_of_tokens and gram not in feature_tokens:
                    feature_tokens['has_NEG_'+gram] = True
        elif len(tokens_ngrams)>0:
            for gram in tokens_ngrams:
                if gram in bag_of_tokens and gram not in feature_tokens:
                    feature_tokens['has_'+gram] = True
    return feature_tokens

def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z

def clean_pos(pos):
    
    pos_tags={'NN':'NN', 'NNP':'NN','NNP-LOC':'NN', 'NNS':'NN', 'JJ':'JJ', 'JJR':'JJ', 'JJS':'JJ', 'RB':'RB', 'RBR':'RB',
     'RBS':'RB', 'VB':'VB', 'VBD':'VB', 'VGB':'VB', 'VBN':'VB', 'VBP':'VB', 'VBZ':'VB'}
    
    if pos in pos_tags:
        pos = pos_tags[pos]
    return pos

def load_nrc_dict(folder=config.DATA_PATH):
    nrc = pd.read_csv(os.path.join(folder, \
                                   'NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt'),
                     delimiter='\t',
                     header=None,
                     names=['word', 'emotion', 'value'])
    
    nrc_d = dict()
    counts = nrc['word'].value_counts()
    for w in counts.index:
        pos, neg = get_score_nrc_load(w, nrc)
        nrc_d[w] = {'pos': pos, 'neg': neg}
    
    return nrc_d

def get_score_nrc_load(word, nrc):
    word_emos = nrc[nrc['word'] == word]
    pos, neg = 0, 0
    if word_emos.shape[0] > 0:
        pos = word_emos[word_emos['emotion']=='positive']['value'].values[0]
        neg = word_emos[word_emos['emotion']=='negative']['value'].values[0]
    return pos, neg

def get_score_nrc(word):
    pos, neg = 0, 0
    sents = nrc.get(word, None)
    if not sents is None:
        pos = sents['pos']
        neg = sents['neg']
    return pos, neg

def get_nrc_feats(tokens):
    tot_pos, tot_neg = 0, 0
    total = 0
    count = len(tokens)
    if count > 0:
        for tok in tokens:
            pos, neg = get_score_nrc(tok)
            tot_pos += pos
            tot_neg += neg
        total = (tot_pos - tot_neg) / count
    return {'nrc_pos': tot_pos,
            'nrc_neg': tot_neg,
             'nrc_total': total}
    
def sentiscore(tokens):
    words_pos = nltk.pos_tag(tokens)
    tokens = {}
    tag_wn={'NN':wn.NOUN,'JJ':wn.ADJ,'VB':wn.VERB,'RB':wn.ADV}
    okays = ['ok','okay']
    for word_pos in words_pos:
        if word_pos[0] in string.punctuation or word_pos[0] in okays:
            continue
        pos = clean_pos(word_pos[1])
        if pos in tag_wn:
            tokens[word_pos[0]] = wn.synsets(word_pos[0],tag_wn[pos])
        else:
            continue
    scores = {}
    scores['total_score']=0
    for word in tokens:
        for synset in tokens[word]:
            if synset is None:
                continue
            info = synset.name().split('.'[0].replace(' ',' '))
            temps = get_score(info[0],info[1])
            positive = 0
            negative = 0
            total_synsets = 0
            if len(temps) != 0:
                for score in temps:
                    positive += score['pos']
                    negative += score['neg']
                    total_synsets +=1
                t_score = positive - negative
            else:
                t_score = 0
            f_score = 'neu'
            if t_score < 0:
                f_score = 'neg'
            elif t_score > 0:
                f_score = 'pos'
            if total_synsets == 0:
                break
            scores[word+'_score'] = (t_score)/total_synsets
            scores[word+'_sentiment'] = f_score
            scores['total_score']+= t_score/total_synsets
            break
    return scores

def get_score(word,pos=None):
    senti_scores = []
    list_pos = [wn.ADJ,wn.ADV,wn.NOUN,wn.VERB]
    if word == 'd' or pos == 'd' or pos not in list_pos:
        return senti_scores
    
    synsets = wn.synsets(word,pos)
    for synset in synsets:
        if (synset.pos(), synset.offset()) in swn.pos_synset:
            pos_val, neg_val = swn.pos_synset[(synset.pos(), synset.offset())]
            senti_scores.append({"pos":pos_val,"neg":neg_val,\
            "obj": 1.0 - (pos_val - neg_val),'synset':synset})

    return senti_scores

def extract_features(bag_of_words,bag_of_lemmas,text):
    

    #ngrams_words,ngrams_lemmas,pos_tagged = extract_ngrams([text])
    
    #feature_lemmas = extract_feature_tokens(ngrams_lemmas,bag_of_lemmas)
    #feature_words = extract_feature_tokens(ngrams_words,bag_of_words)

    feature_set= {}
    
    # POS TAGGED
    #for pos in pos_tagged:
    #    if pos in feature_set:
    #        feature_set[pos]+=1
    #    else:
    #        feature_set[pos]=1
            
    # Number of Hashtags
    feature_set['#']=text.count('#')
        
    # Punctuation mark
    feature_set['!']= text.count('!')
    feature_set['?']= text.count('?')
    if text.endswith('!'):
        feature_set['ends_with_!']=True
    elif text.endswith('?'):
        feature_set['ends_with_?']=True
        
    # Elongated words
    elong = re.compile("([abdfghijkmnostvxyz])\\1{1,}")
    feature_set['words_elong']= sum([1 for token in text.split() if elong.search(token)])
            
    # Caps
    feature_set['words_all_caps']= sum([1 for token in text.split() if token.isupper() and len(token)>=3])
    
    text=re.sub(r'([)(])\1+', r'\1', text)
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens if len(w) >1]
    
    #SentiWordNet
    feature_scores = sentiscore(tokens)
    
    feature_set=merge_two_dicts(feature_set,feature_scores)
    
    #Harvard lexicon
    #hiv4_score = hiv4.get_score(hiv4.tokenize(text))
    #del hiv4_score['Subjectivity']
    #hiv4_score['hiv4_neg'] = hiv4_score['Negative']
    #hiv4_score['hiv4_pos'] = hiv4_score['Positive']
    #hiv4_score['hiv4_pol'] = hiv4_score['Polarity']
    #del hiv4_score['Positive']
    #del hiv4_score['Negative']
    #del hiv4_score['Polarity']
    #
    #feature_set = merge_two_dicts(feature_set, hiv4_score)
    #
    ##NRC lexicon
    #nrc_score = get_nrc_feats(tokens)
    #feature_set = merge_two_dicts(feature_set, nrc_score)
    
    return feature_set

def prepare_data(data):
   
    tweets_data = list()
    for tweet in data:
        text = ' '.join(tweet)
        tweets_data.append(text)
        
    features_list = list()
    count = 0
    bag_of_words = []
    bag_of_lemmas = []
    for tweet in tweets_data:
        features_list.append(extract_features(bag_of_words,bag_of_lemmas,tweet))
        count += 1
        
    return features_list


def download_swn(folder=config.RESOURCES_PATH):
    url = 'http://sentiwordnet.isti.cnr.it/SentiWordNet_3.0.0.tgz'
    name = 'SentiWordNet_3.0.0.tgz'
    path = os.path.join(folder, name)
    filename, _ = urllib.request.urlretrieve(url, path)
    with tarfile.open(os.path.join(folder, name)) as f:
        f.extractall(folder)


logger.debug('Loading stopwords...')
stopwords = load_stopwords()
logger.debug('Downloading SentiWordNet...')
download_swn()
logger.debug('Loading SentiWordNet...')
swn = SentiWordNet(os.path.join(config.RESOURCES_PATH, 'home/swn/www/admin/dump/SentiWordNet_3.0.0_20130122.txt'))
#logger.debug('Loading NRC...')
#nrc = load_nrc_dict()
#logger.debug('Loaded NRC')

