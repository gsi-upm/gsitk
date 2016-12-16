
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

class Twitter_utilities:
	"""This class is aimed to simplify the process when working with tweets. 
	"""

    def __init__(self):
        self.eyes = r"[8:=;]"
        self.nose = r"['`\-]?"
        self.FLAGS = re.MULTILINE | re.DOTALL

    def preprocessor_twitter(self, text):
    	"""This function cleans tweets, deleting typical characters appearing on them which may not give any information
    	"""
        text = re.sub(self.regHttp, '', text)
        text = re.sub(self.regAt, '', text)
        text = re.sub('RT : ', '', text)
        text = re.sub(self.regHttps, '', text)
        text = re.sub('[0-9]', '', text)
        return text.lstrip().strip()

    def hashtag(self,text):
    	"""This functions returns the hashtags appearing on a tweet
    	"""
        text = text.group()
        hashtag_body = text[1:]
        if hashtag_body.isupper():
            result = "<hashtag> {} <allcaps>".format(hashtag_body)
        else:
            result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=self.FLAGS))
        return result

    def allcaps(text):
    	"""This function returns 
    	"""
        text = text.group()
        return text.lower() + " <allcaps>"

    def tokenize(self,text):
    	"""This function tokenizes a tweet
    	"""
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
        
    def twitter_features(self,text):
    	"""Function to extract specific Twitter features
    	"""
        length = float(len(text.split()))
        features = list()

        mention_count = ('@COUNT', float(len(re.findall('@username', text)))/length)
        hashtag_count = ('#COUNT', float(len(re.findall('#', text)))/length)
        rt_count = ('RT', float(len(re.findall('RT @username', text)))/length)
        url_count = ('URL', float(len(re.findall('http[s]?://', text)))/length)
        pic_count = ('PIC', float(len(re.findall('pic.twitter.com', text)))/length)

        features.append(mention_count,hashtag_count,rt_count,url_count,pic_count)
        return features