import unittest
import sentiment_feature_extraction as sfe
import sentiment_feature_extraction_spa as sfespa

class FeatureExtractionTest(unittest.TestCase):

    def find_ngrams_test(self):
        words = ["hello","world","nice","day"]
        bigrams=[('hello', 'world'), ('world', 'nice'), ('nice', 'day')]
        trigrams=[('hello', 'world', 'nice'), ('world', 'nice', 'day')]
        self.assertEqual(sfe.find_ngrams(words,2), bigrams)
        self.assertEqual(sfe.find_ngrams(words,3), trigrams)

    def identify_negations_on_unigrams_test(self):

        sentences = [["I","like","the","sun","but","i","dont","like","the","rain"],
                     ["I","have","no","idea","of","the","solution"],
                    ["No","quiero","ir","al","cine",",","pero","si","al","teatro"]]
        bag_of_words = ["idea","solution","like","sun","rain","quiero","cine","teatro"]
        negations = ["no","dont"]
        stopwords = ["I","of","the","al"]
        features = [{'has_NEG_rain': True, 'has_like': True, 'has_sun': True, 'has_NEG_like': True},
                    {'has_NEG_idea': True, 'has_NEG_solution': True},
                    {'has_cine': True, 'has_teatro': True, 'has_quiero': True}]
        for i in range(len(sentences)):
            self.assertEqual(sfe.identify_negations_on_unigrams(sentences[i],bag_of_words,stopwords,negations),features[i])

    def exclamation_and_interrogation_test(self):
        sentences = ["I have no idea!!!!","What is your question?","Estoy nerviso!! o no?"]
        features = [{'!': 4, 'end_with_!': True, '?': 0},
                    {'!': 0, 'end_with_?': True, '?': 1},
                    {'!': 2, 'end_with_?': True, '?': 1}]
        for i in range(len(sentences)):
            self.assertEqual(sfe.exclamation_and_interrogation(sentences[i]),features[i])

    def emoticons_from_dictionary_test(self):
        sentences = ["No te puedo ayudar :(", "I hope you enjoy your time in Madrid :) :)",":( :)"]
        dict_emoticons = {":)":1,":(":-1}
        features = [{'number_emoticons': 1, 'score_emoticons': -1, 'end_with_negative_emoticon': True, 'score_:(': -1},
                    {'number_emoticons': 2, 'score_emoticons': 2, 'score_:)': 1, 'ends_with_positive_emoticon': True},
                    {'ends_with_positive_emoticon': True, 'number_emoticons': 2, 'score_emoticons': 0, 'score_:)': 1, 'score_:(': -1}]
        for i in range(len(sentences)):
            self.assertEqual(sfe.emoticons_from_dictionary(sentences[i],dict_emoticons),features[i])

    def elongated_words_test(self):
        sentences = ["Largooo de aquiii","I am tireddddd"]
        results= [2,1]
        for i in range(len(sentences)):
            self.assertEqual(sfe.elonganted_words(sentences[i]),results[i])

    def caps_words_test(self):
        sentences = ["i DONT like YOU","Estoy ENFADADO","I love the SUN"]
        results = [2,1,1]
        for i in range(len(sentences)):
            self.assertEqual(sfe.caps_words(sentences[i]),results[i])

    def merge_two_dicts_test(self):
        dict1 = {"a":1,"b":2}
        dict2 = {"c":3,"d":4}
        dict_final = {"a":1,"b":2,"c":3,"d":4}
        self.assertEqual(sfe.merge_two_dicts(dict1,dict2),dict_final)

    def extract_unigrams_pos_test(self):

        sentences = ["I love my new book","This sunset is amazing","that bike is from your brother"]
        stopwords = ["i","my","this","is","that","from","your"]
        results = [(['love', 'new', 'book'], ['VBP', 'JJ', 'NN']),
                    (['sunset', 'amazing'], ['NN', 'VBG']),
                    (['bike', 'brother'], ['NN', 'NN'])]
        for i in range(len(sentences)):
            self.assertEqual(sfe.extract_unigrams_pos(sentences[i],stopwords),results[i])


if __name__ == '__main__':
    unittest.main()
