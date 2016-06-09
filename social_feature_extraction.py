from nltk.stem import SnowballStemmer
import clean_text
import re,pandas,csv
import nltk,nltk.data
from nltk import word_tokenize, FreqDist, pos_tag
import string
import nltk.tokenize as tokenize
import pattern.en, pattern.es
import clean_text as ct
import utilities

class Social_Pscychologist_Features:

    #This function evaluate the pscychologist features from the MRC dictionary
    def identify_pscychologist_feature(features,dictionary_MRC):
        imagenery = 0
        age_of_acquisition = 0
        familiarity = 0
        concreteness = 0
        matches = 0
        results = {}

        for feature in features:
            imagenery += dictionary_MRC[feature]['I']*features[feature]
            age_of_acquisition += dictionary_MRC[feature]['AOA']*features[feature]
            familiarity += dictionary_MRC[feature]['F']*features[feature]
            concreteness += dictionary_MRC[feature]['C']*features[feature]
            matches += features[feature]

        imagenery = float("{0:.2f}".format(round(imagenery / matches,2)))
        age_of_acquisition = float("{0:.2f}".format(round(age_of_acquisition / matches,2)))
        familiarity = float("{0:.2f}".format(round(familiarity / matches,2)))
        concreteness = float("{0:.2f}".format(round(concreteness / matches,2)))

        results['Imagenery'] = imagenery
        results['Age_of_Acquisition'] = age_of_acquisition
        results['Familiarity'] = familiarity
        results['Concreteness'] = concreteness
        return results

    #This function evaluate the NRC hashtag emotion Lexicon
    def emotion_hashtag(selected_features,hashtag_emotion_lexicon):
        emotion_user = {}
        matches = {}
        for word in selected_features:
            for emotion in hashtag_emotion_lexicon[word]:
                if emotion in emotion_user:
                    emotion_user[emotion+'_NRC'] += selected_features[word]*float(hashtag_emotion_lexicon[word][emotion])
                    matches[emotion+'_NRC'] += selected_features[word]
                else:
                    emotion_user[emotion+'_NRC'] = selected_features[word]*float(hashtag_emotion_lexicon[word][emotion])
                    matches[emotion+'_NRC'] = selected_features[word]

        score_emotion = {emotion : emotion_user[emotion]/matches[emotion] for emotion in emotion_user}
        return score_emotion

    # General function to pass from features selected from a dictionary to categories
    def features_to_categories(features,list_categories):
        selected_categories = {}
        for feature in features:
            categories = list_categories[feature]
            for category in categories:
                if category not in selected_categories:
                    selected_categories[category] = features[feature]
                else:
                    selected_categories[category] += features[feature]
        return selected_categories

    # Function to extract the number of words related to boyfriend/girlfriend category
    def bf_gf_word_count (self,text,language):
        if language == 'english':
            bf_words = 'wife|gf|girlfriend|dw'
            gf_words = 'husband|bf|boyfriend|hubby|dh'
        elif language == 'spanish':
            bf_words = 'mujer|novia|esposa'
            gf_words = 'marido|novio|esposo'
        elif language == 'dutch':
            bf_words = 'vrouw|vriendin'
            gf_words = 'man|bf|vriend'

        bf_count = ('GFCOUNT', len(re.findall(bf_words, text, re.I)))
        gf_count = ('BFCOUNT', len(re.findall(gf_words, text, re.I)))

        return bf_count, gf_count


class Negations_Features:
    #This function is going to identify the negations that appear in the set of unigrams,
    #and it is going to negate them in case a negation appear in the set. This ngrams negation is going
    #to appear in all the ngrams until there is a punctuation mark that indicate the end of the negation
    #The result is going to be dictionary with all the matches from the bag of unigrams that is has been passed as
    #parameter, the negation of the ngrams in case there is a negation and the combinations of this unigrams to make
    #bigrams and trigrams

    #Params
    # -tokens, list of word that it is wanted to negate and match with the bag_of_tokens
    # -bag_of_tokens, set of tokens
    # -stopwords, words that are wanted to skip
    # -negations, words that indicate negation
    def identify_negations_on_unigrams(tokens,bag_of_tokens,stopwords,negations):
        feature_tokens = {}
        tokens_ngrams=[]
        auxiliar_tokens=[]
        sent_punctuation = [".",",", ";", "!", "?", ":", "\n", "\r"]
        negation=False
        for token in tokens:
            if token in negations:
                tokens_ngrams += [gram for gram in auxiliar_tokens if gram not in stopwords]
                tokens_ngrams += ['_'.join(bigram) for bigram in utilities.Utilities.find_ngrams(auxiliar_tokens,2)]
                tokens_ngrams += ['_'.join(trigram) for trigram in utilities.Utilities.find_ngrams(auxiliar_tokens,3)]
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
                    tokens_ngrams += ['_'.join(bigram) for bigram in utilities.Utilities.find_ngrams(auxiliar_tokens,2)]
                    tokens_ngrams += ['_'.join(trigram) for trigram in utilities.Utilities.find_ngrams(auxiliar_tokens,3)]
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
            tokens_ngrams += ['_'.join(bigram) for bigram in utilities.Utilities.find_ngrams(auxiliar_tokens,2)]
            tokens_ngrams += ['_'.join(trigram) for trigram in utilities.Utilities.find_ngrams(auxiliar_tokens,3)]
            if len(tokens_ngrams)>0 and negation:
                for gram in tokens_ngrams:
                    if gram in bag_of_tokens and gram not in feature_tokens:
                        feature_tokens['has_NEG_'+gram] = True
            elif len(tokens_ngrams)>0:
                for gram in tokens_ngrams:
                    if gram in bag_of_tokens and gram not in feature_tokens:
                        feature_tokens['has_'+gram] = True
        return feature_tokens


class Emoticons_features:

    #This function returns the number of emoticons, the score that each one has, the total score and if the sentence is
    #finished with a negative, positive or neutral emoticon
    def emoticons_from_dictionary(text,dict_emoticons):
        feature_set = dict()
        feature_set['score_emoticons']=0
        feature_set['number_emoticons']=0
        tweet_tokens_emoticons=clean_text.normalize_emoticons(text)
        text = tweet_tokens_emoticons.split()
        for emoticon in text:
            if emoticon in dict_emoticons:
                feature_set['score_emoticons']+=dict_emoticons[emoticon]
                feature_set['score_'+emoticon]=dict_emoticons[emoticon]
                feature_set['number_emoticons']+=1
                if tweet_tokens_emoticons.endswith(emoticon):
                    if dict_emoticons[emoticon] > 0:
                        feature_set['ends_with_positive_emoticon']=True
                    elif dict_emoticons[emoticon] <0:
                        feature_set['end_with_negative_emoticon']=True
                    else:
                        feature_set['end_with_neutral_emoticon']=True
        return feature_set


