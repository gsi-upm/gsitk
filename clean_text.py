import re
import unicodedata

class CleanText:

    def __init__(self):
        # init params
        self.regHttp = re.compile('(http://)[a-zA-Z0-9]*.[a-zA-Z0-9/]*(.[a-zA-Z0-9]*)?')
        self.regHttps = re.compile('(https://)[a-zA-Z0-9]*.[a-zA-Z0-9/]*(.[a-zA-Z0-9]*)?')
        self.regAt = re.compile('@([a-zA-Z0-9]*[*_/&%#@$]*)*[a-zA-Z0-9]*')
        self.punctuation = [")", "(", "\"", "'", "-","_","+","\\","/"] #punctuation marks to delete
        self.sent_punctuation = [".",",", ";", "!", "?", ":", "\n", "\r"] #punctuation mark meaning the end of a sentence
        self.pos_tags={'NN':'NN', 'NNP':'NN','NNP-LOC':'NN', 'NNS':'NN', 'JJ':'JJ', 'JJR':'JJ', 'JJS':'JJ', 'RB':'RB', 'RBR':'RB',
         'RBS':'RB', 'VB':'VB', 'VBD':'VB', 'VGB':'VB', 'VBN':'VB', 'VBP':'VB', 'VBZ':'VB'}

    ##This function delete all the special characters in short sample of text. It is created focused on Twitter
    def preprocessor_twitter(self, text):
        text = re.sub(self.regHttp, '', text)
        text = re.sub(self.regAt, '', text)
        text = re.sub('RT : ', '', text)
        text = re.sub(self.regHttps, '', text)
        text = re.sub('[0-9]', '', text)
        return text.lstrip().strip()

    ##This function delete the punctuation marks that doesn't mean the end of a sentence.
    ##In case, the punctuation mark means the end of a sentence, it is going to be separated from text with a white space
    ##before and after the punctuation mark.
    def delete_punctuation_marks(self,text):
        text = re.sub(r'#([^\s]+)',r'\1',text)
        text = re.sub(r'([abdefghijkmnopqstuvwyz])\1+', r'\1', text)
        for punct in self.punctuation:
            text = re.sub(re.escape(punct),"",text)
        for punct in self.sent_punctuation:
            text = re.sub(re.escape(punct)," "+punct+" ",text)
        return text.strip()

    #Delete accents from a word in case it is necessary
    def delete_accents(word):
        return unicodedata.normalize('NFKD', unicode(word)).encode('ascii', 'ignore')

    #Normalize emoticons in a text. For example, :)))) = :)
    def normalize_emoticons(self,text):
        text=re.sub(r'([)(])\1+', r'\1', text)
        text=re.sub(r'([:])\1+', r'\1', text)
        text=re.sub(r'(["])','',text)
        return text

    #Normalize the part of speech elements.
    def clean_pos(self,pos):
        if pos in self.pos_tags:
            pos = self.pos_tags[pos]
        return pos