import re
import unicodedata


##This function delete all the special characters in short sample of text. It is created focused on Twitter
def preprocessor_twitter(text):
    #Delete URLs, users, and numbers
    regHttp = re.compile('(http://)[a-zA-Z0-9]*.[a-zA-Z0-9/]*(.[a-zA-Z0-9]*)?')
    regHttps = re.compile('(https://)[a-zA-Z0-9]*.[a-zA-Z0-9/]*(.[a-zA-Z0-9]*)?')
    regAt = re.compile('@([a-zA-Z0-9]*[*_/&%#@$]*)*[a-zA-Z0-9]*')
    text = re.sub(regHttp, '', text)
    text = re.sub(regAt, '', text)
    text = re.sub('RT : ', '', text)
    text = re.sub(regHttps, '', text)
    text = re.sub('[0-9]', '', text)
    return text.lstrip().strip()

##This function delete the punctuation marks that doesn't mean the end of a sentence.
##In case, the punctuation mark means the end of a sentence, it is going to be separated from text with a white space
##before and after the punctuation mark.
def delete_punctuation_marks(text):
    #punctuation marks to delete
    punctuation = [")", "(", "\"", "'", "-","_","+","\\","/"]
    #punctuation mark meaning the end of a sentence
    sent_punctuation = [".",",", ";", "!", "?", ":", "\n", "\r"]
    text = re.sub(r'#([^\s]+)',r'\1',text)
    text = re.sub(r'([abdefghijkmnopqstuvwyz])\1+', r'\1', text)
    for punct in punctuation:
        text = re.sub(re.escape(punct),"",text)
    for punct in sent_punctuation:
        text = re.sub(re.escape(punct)," "+punct+" ",text)
    return text.strip()

#Delete accents from a word in case it is necessary
def delete_accents(word):
    return unicodedata.normalize('NFKD', unicode(word)).encode('ascii', 'ignore')

#Normalize emoticons in a text. For example, :)))) = :)
def normalize_emoticons(text):
    text=re.sub(r'([)(])\1+', r'\1', text)
    text=re.sub(r'([:])\1+', r'\1', text)
    text=re.sub(r'(["])','',text)
    return text

#Normalize the part of speech elements.
def clean_pos(pos):
    pos_tags={'NN':'NN', 'NNP':'NN','NNP-LOC':'NN', 'NNS':'NN', 'JJ':'JJ', 'JJR':'JJ', 'JJS':'JJ', 'RB':'RB', 'RBR':'RB',
     'RBS':'RB', 'VB':'VB', 'VBD':'VB', 'VGB':'VB', 'VBN':'VB', 'VBP':'VB', 'VBZ':'VB'}
    if pos in pos_tags:
        pos = pos_tags[pos]
    return pos