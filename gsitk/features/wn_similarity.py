#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2017 Ganggao Zhu- Grupo de Sistemas Inteligentes
# gzhu[at]dit.upm.es
# DIT, UPM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.corpus.reader.wordnet import information_content

#from sematch.semantic.sparql import EntityFeatures, StatSPARQL
#from sematch.semantic.graph import GraphIC
#from sematch.utility import FileIO, memoized


import math
import sys
from collections import Counter
from collections.abc import Hashable


import json
import os


class FileIO:

    @staticmethod
    def path():
        return os.path.dirname(__file__)

    @staticmethod
    def filename(name):
        if FileIO.path() not in name:
            name = os.path.join(FileIO.path(), name)
        return name

    @staticmethod
    def read_json_file(name):
        data = []
        with open(FileIO.filename(name),'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    @staticmethod
    def save_json_file(name, data):
        with open(FileIO.filename(name), 'w') as f:
            for d in data:
                json.dump(d, f)
                f.write("\n")

    @staticmethod
    def append_json_file(name, data):
        with open(FileIO.filename(name), 'a') as f:
            for d in data:
                json.dump(d, f)
                f.write("\n")

    @staticmethod
    def append_list_file(name, data):
        with open(FileIO.filename(name), 'a') as f:
            for d in data:
                f.write(d)
                f.write('\n')

    @staticmethod
    def save_list_file(name, data):
        with open(FileIO.filename(name),'w') as f:
            for d in data:
                f.write(d)
                f.write('\n')

    @staticmethod
    def read_list_file(name):
        with open(FileIO.filename(name),'r') as f:
            data = [line.strip() for line in f]
        return data


import collections
import functools

class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)

class WordNetSimilarity:

    """Extend the NLTK's WordNet with more similarity metrics, word lemmatization, and multilingual."""

    def __init__(self, ic_corpus='brown'):
        self._ic_corpus = wordnet_ic.ic('ic-brown.dat') if ic_corpus == 'brown' else wordnet_ic.ic('ic-semcor.dat')
        self._wn_max_depth = 19
        self._default_metrics = ['path','lch','wup','li','res','lin','jcn','wpath']
        self._wn_lemma = WordNetLemmatizer()

    def method(self, name):
        def function(syn1, syn2):
            score = getattr(self, name)(syn1, syn2)
            return abs(score)
        return function

    def synset_expand(self, s):
        result = [s]
        hypos = s.hyponyms()
        if not hypos:
            return result
        for h in hypos:
            result.extend(self.synset_expand(h))
        return result

    #return all the noun synsets in wordnet
    def get_all_synsets(self):
        return wn.all_synsets('n')

    def get_all_lemma_names(self):
        return wn.all_lemma_names('n')

    def offset2synset(self, offset):
        '''
        offset2synset('06268567-n')
        Synset('live.v.02')
        '''
        return wn._synset_from_pos_and_offset(str(offset[-1:]), int(offset[:8]))

    def synset2offset(self, ss):
        return '%08d-%s' % (ss.offset(), ss.pos())

    #semcor live%2:43:06::
    def semcor2synset(self, sense):
        return wn.lemma_from_key(sense).synset()

    def semcor2offset(self, sense):
        '''
        semcor2synset('editorial%1:10:00::')
        06268567-n
        '''
        return self.synset2offset(self.semcor2synset(sense))

    def word2synset(self, word, pos=wn.NOUN):
        word = self._wn_lemma.lemmatize(word)
        return wn.synsets(word, pos)

    def languages(self, l=None):
        """Return a list of supported languages or find the corresponding language code of supported language.

        :param l: The default value is None or the name of language
        :return: if the default none is set, return a list of supported language names. When l is assigned with a
        language name, the corresponding code is returned.

        User should use this function to check the languages and find the language code.
        """
        langs = {'albanian':'als',
                 'arabic':'arb',
                 'bulgarian':'bul',
                 'chinese_simplified':'cmn',
                 'chinese_traditional':'qcn',
                 'danish':'dan',
                 'greek':'ell',
                 'english':'eng',
                 'persian':'fas',
                 'finnish':'fin',
                 'french':'fra',
                 'hebrew':'heb',
                 'croatian':'hrv',
                 'icelandic':'isl',
                 'italian':'ita',
                 'japanese':'jpn',
                 'catalan':'cat',
                 'basque':'eus',
                 'galicain':'glg',
                 'spanish':'spa',
                 'indonesian':'ind',
                 'malay':'zsm',
                 'dutch':'nld',
                 'polish':'pol',
                 'portuguese':'por',
                 'romanian':'ron',
                 'lithuanian':'lit',
                 'slovak':'slk',
                 'slovene':'slv',
                 'swedish':'swe',
                 'thai':'tha'}
        if l:
            if l.lower() in langs:
                return langs[l.lower()]
            else:
                return l+" is not supported!"
        return map(lambda x:x.capitalize(), langs.keys())


    def multilingual2synset(self, word, lang='spa'):
        """
        Map words in different language to wordnet synsets
        ['als', 'arb', 'cat', 'cmn', 'dan', 'eng', 'eus', 'fas', 'fin', 'fra', 'fre',
         'glg', 'heb', 'ind', 'ita', 'jpn', 'nno','nob', 'pol', 'por', 'spa', 'tha', 'zsm']
        :param word: a word in different language that has been defined in
         Open Multilingual WordNet, using ISO-639 language codes.
        :param lang: the language code defined
        :return: wordnet synsets.
        """
        if sys.version_info[0] < 3:
            return wn.synsets(word.decode('utf-8'), lang=lang, pos=wn.NOUN)
        return wn.synsets(word, lang=lang, pos=wn.NOUN)


    @memoized
    def similarity(self, c1, c2, name='wpath'):
        """
        Compute semantic similarity between two concepts
        :param c1:
        :param c2:
        :param name:
        :return:
        """
        return self.method(name)(c1, c2)

    def max_synset_similarity(self, syns1, syns2, sim_metric):
        """
        Compute the maximum similarity score between two list of synsets
        :param syns1: synset list
        :param syns2: synset list
        :param sim_metric: similarity function
        :return: maximum semantic similarity score
        """
        return max([sim_metric(c1, c2) for c1 in syns1 for c2 in syns2] + [0])

    @memoized
    def word_similarity(self, w1, w2, name='wpath'):
        """ Return similarity score between two words based on WordNet.

        :param w1: first word to be compared which should be contained in WordNet
        :param w2: second word to be compared which should be contained in WordNet
        :param name: the name of knowledge-based semantic similarity metrics
        :return: numerical score indicating degree of similarity between two words. The
        minimum score is 0. If one of the input words is not contained in WordNet, 0 is given. The up bound of
        the similarity score depends on the similarity metric you use. Bigger similarity values indicate higher
        similarity between two words.
        :rtype : Float

        """
        s1 = self.word2synset(w1)
        s2 = self.word2synset(w2)
        sim_metric = lambda x, y: self.similarity(x, y, name)
        return self.max_synset_similarity(s1, s2, sim_metric)

    @memoized
    def best_synset_pair(self, w1, w2, name='wpath'):
        s1 = self.word2synset(w1)
        s2 = self.word2synset(w2)
        sims = Counter({(c1, c2):self.similarity(c1, c2, name) for c1 in s1 for c2 in s2})
        return sims.most_common(1)[0][0]

    def word_similarity_all_metrics(self, w1, w2):
        return {m:self.word_similarity(w1, w2, name=m) for m in self._default_metrics}

    @memoized
    def word_similarity_wpath(self, w1, w2, k):
        s1 = self.word2synset(w1)
        s2 = self.word2synset(w2)
        sim_metric = lambda x, y: self.wpath(x, y, k)
        return self.max_synset_similarity(s1, s2, sim_metric)

    @memoized
    def monol_word_similarity(self, w1, w2, lang='spa', name='wpath'):
        """
         Compute mono-lingual word similarity, two words are in same language.
        :param w1: word
        :param w2: word
        :param lang: language code
        :param name: name of similarity metric
        :return: semantic similarity score
        """
        s1 = self.multilingual2synset(w1, lang)
        s2 = self.multilingual2synset(w2, lang)
        sim_metric = lambda x, y: self.similarity(x, y, name)
        return self.max_synset_similarity(s1, s2, sim_metric)

    @memoized
    def crossl_word_similarity(self, w1, w2, lang1='spa', lang2='eng', name='wpath'):
        """
         Compute cross-lingual word similarity, two words are in different language.
        :param w1: word
        :param w2: word
        :param lang1: language code for word1
        :param lang2: language code for word2
        :param name: name of similarity metric
        :return: semantic similarity score
        """
        s1 = self.multilingual2synset(w1, lang1)
        s2 = self.multilingual2synset(w2, lang2)
        sim_metric = lambda x, y: self.similarity(x, y, name)
        return self.max_synset_similarity(s1, s2, sim_metric)

    def least_common_subsumer(self, c1, c2):
        return c1.lowest_common_hypernyms(c2)[0]

    def synset_ic(self, c):
        return information_content(c, self._ic_corpus)

    def dpath(self, c1, c2, alpha=1.0, beta=1.0):
        lcs = self.least_common_subsumer(c1, c2)
        path = c1.shortest_path_distance(c2)
        path = 1.0 / (1 + path)
        path = path**alpha
        depth = lcs.max_depth() + 1
        depth = depth*1.0/(1 + self._wn_max_depth)
        depth = depth**beta
        return math.log(1+path*depth,2)

    def wpath(self, c1, c2, k=0.8):
        lcs = self.least_common_subsumer(c1,c2)
        path = c1.shortest_path_distance(c2)
        weight = k ** self.synset_ic(lcs)
        return 1.0 / (1 + path*weight)

    def li(self, c1, c2, alpha=0.2,beta=0.6):
        path = c1.shortest_path_distance(c2)
        lcs = self.least_common_subsumer(c1, c2)
        depth = lcs.max_depth()
        x = math.exp(-alpha*path)
        y = math.exp(beta*depth)
        z = math.exp(-beta*depth)
        a = y - z
        b = y + z
        return x * (a/b)

    def path(self, c1, c2):
        return c1.path_similarity(c2)

    def wup(self, c1, c2):
        return c1.wup_similarity(c2)

    def lch(self, c1, c2):
        return c1.lch_similarity(c2)

    def res(self, c1, c2):
        return c1.res_similarity(c2, self._ic_corpus)

    def jcn(self, c1, c2):
        lcs = self.least_common_subsumer(c1, c2)
        c1_ic = self.synset_ic(c1)
        c2_ic = self.synset_ic(c2)
        lcs_ic = self.synset_ic(lcs)
        diff = c1_ic + c2_ic - 2*lcs_ic
        return 1.0/(1 + diff)

    def lin(self, c1, c2):
        return c1.lin_similarity(c2, self._ic_corpus)


class YagoTypeSimilarity(WordNetSimilarity):

    """Extend the WordNet synset to linked data through YAGO mappings"""

    def __init__(self, graph_ic='models/yago_type_ic.txt', mappings="models/type-linkings.txt"):
        WordNetSimilarity.__init__(self)
        self._graph_ic = GraphIC(graph_ic)
        self._mappings = FileIO.read_json_file(mappings)
        self._id2mappings = {data['offset']: data for data in self._mappings}
        self._yago2id = {data['yago_dbpedia']: data['offset'] for data in self._mappings}

    def synset2id(self, synset):
        return str(synset.offset() + 100000000)

    def id2synset(self, offset):
        x = offset[1:]
        return wn._synset_from_pos_and_offset('n', int(x))

    def synset2mapping(self, synset, key):
        mapping_id = self.synset2id(synset)
        if mapping_id in self._id2mappings:
            mapping = self._id2mappings[mapping_id]
            return mapping[key] if key in mapping else None
        else:
            return None

    def synset2yago(self, synset):
        return self.synset2mapping(synset,'yago_dbpedia')

    def synset2dbpedia(self, synset):
        return self.synset2mapping(synset, 'dbpedia')

    def yago2synset(self, yago):
        if yago in self._yago2id:
            return self.id2synset(self._yago2id[yago])
        return None

    def word2dbpedia(self, word):
        return [self.synset2dbpedia(s) for s in self.word2synset(word) if self.synset2dbpedia(s)]

    def word2yago(self, word):
        return [self.synset2yago(s) for s in self.word2synset(word) if self.synset2yago(s)]

    def yago_similarity(self, yago1, yago2, name='wpath'):
        """
        Compute semantic similarity of two yago concepts by mapping concept uri to wordnet synset.
        :param yago1: yago concept uri
        :param yago2: yago concept uri
        :param name: name of semantic similarity metric
        :return: semantic similarity score if both uri can be mapped to synsets, otherwise 0.
        """
        s1 = self.yago2synset(yago1)
        s2 = self.yago2synset(yago2)
        if s1 and s2:
            return self.similarity(s1, s2, name)
        return 0.0

    def word_similarity_wpath_graph(self, w1, w2, k):
        s1 = self.word2synset(w1)
        s2 = self.word2synset(w2)
        return max([self.wpath_graph(c1, c2, k) for c1 in s1 for c2 in s2] + [0])

    def res_graph(self, c1, c2):
        lcs = self.least_common_subsumer(c1,c2)
        yago = self.synset2yago(lcs)
        return self._graph_ic.concept_ic(yago)

    def lin_graph(self, c1, c2):
        lcs = self.least_common_subsumer(c1,c2)
        yago_c1 = self.synset2yago(c1)
        yago_c2 = self.synset2yago(c2)
        yago_lcs = self.synset2yago(lcs)
        lcs_ic = self._graph_ic.concept_ic(yago_lcs)
        c1_ic = self._graph_ic.concept_ic(yago_c1)
        c2_ic = self._graph_ic.concept_ic(yago_c2)
        combine = c1_ic + c2_ic
        if c1_ic == 0.0 or c2_ic == 0.0:
            return 0.0
        return 2.0 * lcs_ic / combine

    def jcn_graph(self, c1, c2):
        lcs = self.least_common_subsumer(c1,c2)
        yago_c1 = self.synset2yago(c1)
        yago_c2 = self.synset2yago(c2)
        yago_lcs = self.synset2yago(lcs)
        lcs_ic = self._graph_ic.concept_ic(yago_lcs)
        c1_ic = self._graph_ic.concept_ic(yago_c1)
        c2_ic = self._graph_ic.concept_ic(yago_c2)
        lcs_ic = 2.0 * lcs_ic
        if c1_ic == 0.0 or c2_ic == 0.0:
            return 0.0
        return 1.0 / 1+(c1_ic + c2_ic - lcs_ic)

    def wpath_graph(self, c1, c2, k=0.9):
        lcs = self.least_common_subsumer(c1, c2)
        path = c1.shortest_path_distance(c2)
        yago_lcs = self.synset2yago(lcs)
        weight = k ** self._graph_ic.concept_ic(yago_lcs)
        return 1.0 / (1 + path*weight)


