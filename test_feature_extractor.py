import unittest
from tools.utilities import find_ngrams, merge_two_dicts
from tools.social_feature_extractor import identify_negations_on_unigrams, emoticons_from_dictionary
from tools.feature_extractor import exclamation_and_interrogation, elonganted_words, caps_words, extract_unigrams_pos

__author__ = 'NachoCP'


class FeatureExtractionTest(unittest.TestCase):

    def test_find_ngrams(self):
        words = ["hello", "world", "nice", "day"]
        bigrams = [('hello', 'world'), ('world', 'nice'), ('nice', 'day')]
        trigrams = [('hello', 'world', 'nice'), ('world', 'nice', 'day')]
        self.assertEqual(find_ngrams(words, 2), bigrams)
        self.assertEqual(find_ngrams(words, 3), trigrams)

    def test_identify_negations_on_unigrams(self):

        sentences = [["I", "like", "the", "sun", "but", "i", "dont", "like", "the", "rain"],
                     ["I", "have", "no", "idea", "of", "the", "solution"],
                     ["No", "quiero", "ir", "al", "cine", ",", "pero", "si", "al", "teatro"]]
        bag_of_words = ["idea", "solution", "like", "sun", "rain", "quiero", "cine", "teatro"]
        negations = ["no", "dont"]
        stopwords = ["I", "of", "the", "al"]
        features = [{'has_NEG_rain': True, 'has_like': True, 'has_sun': True, 'has_NEG_like': True},
                    {'has_NEG_idea': True, 'has_NEG_solution': True},
                    {'has_cine': True, 'has_teatro': True, 'has_quiero': True}]
        for i in range(len(sentences)):
            self.assertEqual(identify_negations_on_unigrams(sentences[i], bag_of_words, stopwords, negations),
                             features[i])

    def test_exclamation_and_interrogation(self):
        sentences = ["I have no idea!!!!", "What is your question?", "Estoy nerviso!! o no?"]
        features = [{'!': 4, 'end_with_!': True, '?': 0},
                    {'!': 0, 'end_with_?': True, '?': 1},
                    {'!': 2, 'end_with_?': True, '?': 1}]
        for i in range(len(sentences)):
            self.assertEqual(exclamation_and_interrogation(sentences[i]), features[i])

    def test_emoticons_from_dictionary(self):
        sentences = ["No te puedo ayudar :(", "I hope you enjoy your time in Madrid :) :)", ":( :)"]
        dict_emoticons = {":)": 1, ":(": -1}
        features = [{'number_emoticons': 1, 'score_emoticons': -1, 'end_with_negative_emoticon': True, 'score_:(': -1},
                    {'number_emoticons': 2, 'score_emoticons': 2, 'score_:)': 1, 'ends_with_positive_emoticon': True},
                    {'ends_with_positive_emoticon': True, 'number_emoticons': 2, 'score_emoticons': 0, 'score_:)': 1,
                     'score_:(': -1}]
        for i in range(len(sentences)):
            self.assertEqual(emoticons_from_dictionary(sentences[i], dict_emoticons), features[i])

    def test_elongated_words(self):
        sentences = ["Largooo de aquiii", "I am tireddddd"]
        results = [2, 1]
        for i in range(len(sentences)):
            self.assertEqual(elonganted_words(sentences[i]), results[i])

    def test_caps_words(self):
        sentences = ["i DONT like YOU", "Estoy ENFADADO", "I love the SUN"]
        results = [2, 1, 1]
        for i in range(len(sentences)):
            self.assertEqual(caps_words(sentences[i]), results[i])

    def test_merge_two_dicts(self):
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}
        dict_final = {"a": 1, "b": 2, "c": 3, "d": 4}
        self.assertEqual(merge_two_dicts(dict1, dict2), dict_final)

    def test_extract_unigrams_pos(self):

        sentences = ["I love my new book", "This sunset is amazing", "that bike is from your brother"]
        stopwords = ["i", "my", "this", "is", "that", "from", "your"]
        results = [(['love', 'new', 'book'], ['VBP', 'JJ', 'NN']),
                   (['sunset', 'amazing'], ['NN', 'VBG']),
                   (['bike', 'brother'], ['NN', 'NN'])]
        for i in range(len(sentences)):
            self.assertEqual(extract_unigrams_pos(sentences[i], stopwords), results[i])
