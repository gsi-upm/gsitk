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

class GeneralTextFeatures:

    #This function identifies the number of exclamation and interrogation signs that appear in a text and if the text ends
    #with one of them.
    def exclamation_and_interrogation(self,text):
        dictionary=dict()
        dictionary['!'] = text.count('!')
        dictionary['?'] = text.count('?')
        if text.endswith('!'):
            dictionary['end_with_!'] = True
        elif text.endswith('?'):
            dictionary['end_with_?'] = True
        return dictionary

    def twitter_features(self,text):
        length = float(len(text.split()))
        features = list()

        mention_count = ('@COUNT', float(len(re.findall('@username', text)))/length)
        hashtag_count = ('#COUNT', float(len(re.findall('#', text)))/length)
        rt_count = ('RT', float(len(re.findall('RT @username', text)))/length)
        url_count = ('URL', float(len(re.findall('http[s]?://', text)))/length)
        pic_count = ('PIC', float(len(re.findall('pic.twitter.com', text)))/length)

        features.append(mention_count,hashtag_count,rt_count,url_count,pic_count)
        return features

   #This function returns the number of words that are elongated in a text. E.g 'holaaaaaaaa'
    #It is created for a spanish text, because it doesn't count the letters 'r' and 'l'.
    def elonganted_words(self,text):
        elong = re.compile("([abdfghijkmnostvxyz])\\1{1,}")
        return sum([1 for token in text.split() if elong.search(token)])

    #This function returns the number of words that are caps in a text
    def caps_words(self,text):
        return sum([1 for token in text.split() if token.isupper() and len(token)>=3])

    def text_word_avglength(self,text):
        length = float(len(text.split()))

        avg_text_length = ('TEXTLEN', length/len(text.split('\n')))
        words_length = 0
        for word in text.split():
            words_length += len(word)
        avg_word_length = ('WORDLEN', words_length/length)

        return avg_text_length,avg_word_length

    # This function tags the POS (parts-of-speech) that appear in the input text and returns counts
    # for each type and text.
    # Input: JSON file {text="sample text",gender="M/F"}
    # Output: list of (POS, count)

    def count_pos(input, language):
        if language == 'english-nltk':
            words = word_tokenize(input)
            pos = pos_tag(words)

        elif language == 'english':
            s = pattern.en.parsetree(input, relations=True, lemmata=True)
            words = []
            pos = []
            for sentence in s:
                for w in sentence.words:
                    words.append(w.string)
                    pos.append((w.string, ct.CleanText.clean_pos(w.type)))

        elif language == 'spanish':
            s = pattern.es.parsetree(input, relations=True, lemmata=True)
            words = []
            pos = []
            for sentence in s:
                for w in sentence.words:
                    words.append(w.string)
                    pos.append((w.string, ct.CleanText.clean_pos(w.type)))

        elif language == 'dutch':
            words = word_tokenize(input, 'dutch')
            tagger = nltk.data.load('taggers/alpino_aubt.pickle')
            pos = tagger.tag(words)

        tags = FreqDist(tag for (word, tag) in pos)
        relative_frequency = []
        for item in tags.items():
            relative_frequency.append((item[0], float(item[1])/tags.N()))
        return relative_frequency

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

    #General function to count occurrences from a bag_of_features (dictionary)
    def extract_features_text(text,bag_of_features):
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

class Twitter_tokenizer:

    def __init__(self):
        self.eyes = r"[8:=;]"
        self.nose = r"['`\-]?"
        self.FLAGS = re.MULTILINE | re.DOTALL

    def hashtag(self,text):
        text = text.group()
        hashtag_body = text[1:]
        if hashtag_body.isupper():
            result = "<hashtag> {} <allcaps>".format(hashtag_body)
        else:
            result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=self.FLAGS))
        return result

    def allcaps(text):
        text = text.group()
        return text.lower() + " <allcaps>"

    def tokenize(self,text):
        def re_sub(pattern, repl):
            return re.sub(pattern, repl, text, flags=self.FLAGS)

        text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
        text = re_sub(r"/"," / ")
        text = re_sub(r"@\w+", "<user>")
        text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(self.eyes, self.nose, self.nose, self.eyes), "<smile>")
        text = re_sub(r"{}{}p+".format(self.eyes, self.nose), "<lolface>")
        text = re_sub(r"{}{}\(+|\)+{}{}".format(self.eyes, self.nose, self.nose, self.eyes), "<sadface>")
        text = re_sub(r"{}{}[\/|l*]".format(self.eyes, self.nose), "<neutralface>")
        text = re_sub(r"<3","<heart>")
        text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
        text = re_sub(r"#\S+", 'hashtag')
        text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
        text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
        text = re_sub(r"([A-Z]){2,}", 'allcaps')

        return text.lower()


