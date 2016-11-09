import re
import nltk
import nltk.data
import string
import nltk.tokenize as tokenize
import pattern
from nltk import word_tokenize, FreqDist, pos_tag
from tools.text_cleaner import TextCleaner

__author__ = 'NachoCP'


def exclamation_and_interrogation(text):
    """
    This function identifies the number of exclamation and interrogation signs that appear in a text and if the text
    ends with one of them.
    :param text:
    :return:
    """
    dictionary = dict()
    dictionary['!'] = text.count('!')
    dictionary['?'] = text.count('?')
    if text.endswith('!'):
        dictionary['end_with_!'] = True
    elif text.endswith('?'):
        dictionary['end_with_?'] = True
    return dictionary


def elonganted_words(text):
    """
    This function returns the number of words that are elongated in a text. E.g 'holaaaaaaaa'
    It is created for a spanish text, because it doesn't count the letters 'r' and 'l'.
    :param text:
    :return:
    """
    elong = re.compile("([abdfghijkmnostvxyz])\\1{1,}")
    return sum([1 for token in text.split() if elong.search(token)])


def caps_words(text):
    """
    This function returns the number of words that are caps in a text
    :param text:
    :return:
    """
    return sum([1 for token in text.split() if token.isupper() and len(token) >= 3])


def text_word_avg_length(text):
    length = float(len(text.split()))

    avg_text_length = ('TEXTLEN', length/len(text.split('\n')))
    words_length = 0
    for word in text.split():
        words_length += len(word)
    avg_word_length = ('WORDLEN', words_length/length)

    return avg_text_length, avg_word_length


def count_pos(input_string, language):
    """
    This function tags the POS (parts-of-speech) that appear in the input text and returns counts
    for each type and text.
    :param input_string: JSON file {text="sample text",gender="M/F"}
    :param language: list of (POS, count)
    :return:
    """
    text_cleaner = TextCleaner()
    if language == 'english-nltk':
        words = word_tokenize(input_string)
        pos = pos_tag(words)

    elif language == 'english':
        s = pattern.en.parsetree(input_string, relations=True, lemmata=True)
        words = []
        pos = []
        for sentence in s:
            for w in sentence.words:
                words.append(w.string)
                pos.append((w.string, text_cleaner.clean_pos(w.type)))

    elif language == 'spanish':
        s = pattern.es.parsetree(input_string, relations=True, lemmata=True)
        words = []
        pos = []
        for sentence in s:
            for w in sentence.words:
                words.append(w.string)
                pos.append((w.string, text_cleaner.clean_pos(w.type)))

    elif language == 'dutch':
        words = word_tokenize(input_string, 'dutch')
        tagger = nltk.data.load('taggers/alpino_aubt.pickle')
        pos = tagger.tag(words)

    tags = FreqDist(tag for (word, tag) in pos)
    relative_frequency = []
    for item in tags.items():
        relative_frequency.append((item[0], float(item[1])/tags.N()))
    return relative_frequency


def extract_unigrams_pos(text, stopwords):
    """
    This function extract unigrams and pos tagged information from a text, it is focused on English
    :param text:
    :param stopwords:
    :return:
    """
    unigrams_words = []
    pos_tagged = []
    tokens = tokenize.word_tokenize(text.lower())
    words_pos = nltk.pos_tag(tokens)
    for word_pos in words_pos:
        if word_pos[0] not in stopwords and word_pos[0] not in string.punctuation:
            unigrams_words.append(word_pos[0].lower())
            pos_tagged.append(word_pos[1])

    return unigrams_words, pos_tagged


def extract_features_text(text, bag_of_features):
    """
    General function to count occurrences from a bag_of_features (dictionary)
    :param text:
    :param bag_of_features:
    :return:
    """
    selected_features = {}
    for feature in bag_of_features:
        ocurrences = text.count(feature)
        if ocurrences == 0:
            continue
        if feature not in selected_features:
            selected_features[feature] = ocurrences
        else:
            selected_features[feature] += ocurrences
    return selected_features
