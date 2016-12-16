import re
import unicodedata

class CleanText:
    """This class is aimed to clean general texts.
    """

    def __init__(self):
        """Init params
        """
        self.regHttp = re.compile('(http://)[a-zA-Z0-9]*.[a-zA-Z0-9/]*(.[a-zA-Z0-9]*)?')
        self.regHttps = re.compile('(https://)[a-zA-Z0-9]*.[a-zA-Z0-9/]*(.[a-zA-Z0-9]*)?')
        self.regAt = re.compile('@([a-zA-Z0-9]*[*_/&%#@$]*)*[a-zA-Z0-9]*')
        self.punctuation = [")", "(", "\"", "'", "-","_","+","\\","/"] #punctuation marks to delete
        self.sent_punctuation = [".",",", ";", "!", "?", ":", "\n", "\r"] #punctuation mark meaning the end of a sentence
        self.pos_tags={'NN':'NN', 'NNP':'NN','NNP-LOC':'NN', 'NNS':'NN', 'JJ':'JJ', 'JJR':'JJ', 'JJS':'JJ', 'RB':'RB', 'RBR':'RB',
         'RBS':'RB', 'VB':'VB', 'VBD':'VB', 'VGB':'VB', 'VBN':'VB', 'VBP':'VB', 'VBZ':'VB'}


    ##This function delete the punctuation marks that doesn't mean the end of a sentence.
    ##In case, the punctuation mark means the end of a sentence, it is going to be separated from text with a white space
    ##before and after the punctuation mark.
    def delete_punctuation_marks(self,text):
        """This function delete the punctuation marks that doesn't mean the end of a sentence.
           In case, the punctuation mark means the end of a sentence, it is going to be separated from text with a white space
           before and after the punctuation mark.
        """
        text = re.sub(r'#([^\s]+)',r'\1',text)
        text = re.sub(r'([abdefghijkmnopqstuvwyz])\1+', r'\1', text)
        for punct in self.punctuation:
            text = re.sub(re.escape(punct),"",text)
        for punct in self.sent_punctuation:
            text = re.sub(re.escape(punct)," "+punct+" ",text)
        return text.strip()

    
    def delete_accents(word):
        """Delete accents from a word in case it is necessary
        """
        return unicodedata.normalize('NFKD', unicode(word)).encode('ascii', 'ignore')

    
    def normalize_emoticons(self,text):
        """Normalize emoticons in a text. For example, :)))) = :)
        """
        text=re.sub(r'([)(])\1+', r'\1', text)
        text=re.sub(r'([:])\1+', r'\1', text)
        text=re.sub(r'(["])','',text)
        return text

    def clean_pos(self,pos):
        """Normalize the part of speech elements.
        """
        if pos in self.pos_tags:
            pos = self.pos_tags[pos]
        return pos