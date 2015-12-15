from nltk.stem import SnowballStemmer
import clean_text
import re
import nltk
import string
import nltk.tokenize as tokenize

#Combines the different input_list to create a set of Ngrams.
#Params
#   -input_list, list of string that it is wanted to combine
#   -n, number of times that it is wanted to combine, e.g: for bigrams 2, for trigrams 3.

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


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


#This function identifies the number of exclamation and interrogation signs that appear in a text and if the text ends
#with one of them.
def exclamation_and_interrogation(text):
    dictionary=dict()
    dictionary['!'] = text.count('!')
    dictionary['?'] = text.count('?')
    if text.endswith('!'):
        dictionary['end_with_!'] = True
    elif text.endswith('?'):
        dictionary['end_with_?'] = True
    return dictionary


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

#This function returns the number of words that are elongated in a text. E.g 'holaaaaaaaa'
#It is created for a spanish text, because it doesn't count the letters 'r' and 'l'.
def elonganted_words(text):
    elong = re.compile("([abdfghijkmnostvxyz])\\1{1,}")
    return sum([1 for token in text.split() if elong.search(token)])

#This function returns the number of words that are caps in a text
def caps_words(text):
    return sum([1 for token in text.split() if token.isupper() and len(token)>=3])

def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z
#This function extract unigrams and pos tagged information from a text, it is focused on English
def extract_unigrams_pos(text,stopwords):
    unigrams_words = []
    pos_tagged = []
    tokens = tokenize.word_tokenize(text.lower())
    words_pos = nltk.pos_tag(tokens)
    for word_pos in words_pos:
        if word_pos[0] not in stopwords and word_pos[0] not in string.punctuation:
            unigrams_words.append(word_pos[0].lower())
            pos_tagged.append(word_pos[1])

    return unigrams_words,pos_tagged
