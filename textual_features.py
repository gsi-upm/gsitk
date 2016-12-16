import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim


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


class TextualFeatures:
	
	def getParagraphCount(text):
		"""Returns the paragraphs contained in a text
		"""
    	return(1 + len(re.findall( r'\n{2,}', text)))

    
    def getWordCount(text):
    	"""Returns the number of words in a text
    	"""
    	return(len(re.findall( r'\w{1,}', text)))

    def getCapsCount(text):
    	"""Returns the number of capital letters in a text
    	"""
        caps = 0
	    all_words = re.findall( r'\w{1,}', text)
	    for word in all_words:
	        a = re.match(r'[ABCDEFGHIJKLMNOPQRSTUVWXYZ]{1,}', word)
	        if  a != None:
	            caps += 1
	    return(caps)

	def getCapsPercentage(text):
		"""Returns the percentage of capital letters with respect to the number of words
		"""
	    if get_number_of_words == 0:
	        return 0
	    return(100*(get_number_of_caps(text))/get_number_of_words(text))

	def getWordsPerParagraph(text):
		"""Returns the average number of words per paragraph
		"""
	    all_words = re.findall( r'\w{1,}', text)
	    if len(all_words) == 0:
	        return(0)
	    paragraphs = text.split("\n\n")
	    percentages = []
	    var = 0
	    med = len(all_words)/len(paragraphs)
	    for par in paragraphs:
	        words_in_par = re.findall( r'\w{1,}', par)
	        percentages.append(100*(len(words_in_par)/len(all_words)))
	        var += (len(words_in_par) - med)**2
	    varianza = (var/len(paragraphs))
	    return(med)	

	def getWordsPerSentence(text):
		"""Returns the number of words per sentence
		"""
	    all_words = re.findall( r'\w{1,}', text)
	    sentences = text.split(".")
	    return(len(all_words)/len(sentences))	

	def getPunctuationPercentage(text):
		"""Returns the percentage of punctuation marks with respect with the number of words
		"""
	    all_words = re.findall( r'\w{1,}', text)
	    p = re.findall( r'[,$.!?:;]', text)
	    if len(all_words) == 0:
	        return(0)
	    return(len(p)/len(all_words))

	def containsWord(text, word):
		"""Returns True if the text contains the word passed as parameter
		"""
	    if len(re.findall( word, text) > 0 ):
	    	return(True)
	    else:
	    	return(False)



	def getCosineSimilarityMatrix(texts):
		"""Returns the cosine similarity matrix of an array of texts
		"""
		tfidf_vectorizer = TfidfVectorizer()
	    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
	    return(cosine_similarity(tfidf_matrix))

	def getWeekday(input_date):
		"""Returns the day of the week from a string formatted as  YYYY/MM/DD
		"""
	    #Class date struct =>  year/month/day
	    #Python date struct => year/month/day
	    date_1 = re.findall(r'(\d{4})-(\d{2})-(\d{2})', input_date)
	    if len(date_1) == 0:
	        return(7)
	    if len(date_1[0]) == 0:
	        return(7)
	    date_2 = date(int(date_1[0][0]),int(date_1[0][1]),int(date_1[0][2]))
	    return(date_2.weekday()) 

	
	def getPOSFeatures(text):
		"""Returns features related to the most common POS tags in an array form
		"""
	    features = [0,0,0,0,0,0,0,0,0,0,0]
	    words = nltk.word_tokenize(text)
	    tags = nltk.pos_tag(words)
	    
	    ind = 0
	    for tag in ['CC', 'CD', 'DT', 'IN', 'JJ', 'NNP', 'VERB', 'NN', 'PRP$', 'RBR', 'VB']:
	        for (word, tag2) in tags:
	            if tag == tag2:
	                features[ind] += 1
	            elif (tag == "VB") & (len(re.findall(tag,tag2)) > 0):
	                features[ind] += 1
	        ind += 1
	    return features


    def exclamation_and_interrogation(self,text):
    	"""This function identifies the number of exclamation and interrogation signs that appear in a text and if the text ends
    		with one of them.
    	"""
        dictionary=dict()
        dictionary['!'] = text.count('!')
        dictionary['?'] = text.count('?')
        if text.endswith('!'):
            dictionary['end_with_!'] = True
        elif text.endswith('?'):
            dictionary['end_with_?'] = True
        return dictionary

       
    def elonganted_words(self,text):
    	"""This function returns the number of words that are elongated in a text. E.g 'holaaaaaaaa'
    		It is created for a spanish text, because it doesn't count the letters 'r' and 'l'.
    	"""
        elong = re.compile("([abdfghijkmnostvxyz])\\1{1,}")
        return sum([1 for token in text.split() if elong.search(token)])


    def text_word_avglength(self,text):
    	"""Returns the average length of the words appearing on a text
    	"""
        length = float(len(text.split()))

        avg_text_length = ('TEXTLEN', length/len(text.split('\n')))
        words_length = 0
        for word in text.split():
            words_length += len(word)
        avg_word_length = ('WORDLEN', words_length/length)

        return avg_text_length,avg_word_length



    def count_pos(input, language):
    	"""This function tags the POS (parts-of-speech) that appear in the input text and returns counts
    		for each type and text.
    		Input: JSON file {text="sample text",gender="M/F"}
    		Output: list of (POS, count)
    	"""
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

    
    def extract_unigrams_pos(text,stopwords):
    	"""This function extract unigrams and pos tagged information from a text, it is focused on English
    	"""
        unigrams_words = []
        pos_tagged = []
        tokens = tokenize.word_tokenize(text.lower())
        words_pos = nltk.pos_tag(tokens)
        for word_pos in words_pos:
            if word_pos[0] not in stopwords and word_pos[0] not in string.punctuation:
                unigrams_words.append(word_pos[0].lower())
                pos_tagged.append(word_pos[1])

        return unigrams_words,pos_tagged

    
    def extract_features_text(text,bag_of_features):
    	"""General function to count occurrences from a bag_of_features (dictionary)
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






class TextualModels:

	def createTfidf(self, texts, ngrams, stop_words_lan, return_model = False):
		"""Creates Tfidf from array of texts. If return_model = True, the model will be returned to operate over it. Else, the Tfidf of the texts will be returned
		"""
		countVect = CountVectorizer(ngram_range=(1, ngrams), stop_words=stop_words_lan)
		counts = countVect.fit_transform(texts)
		tfidfTransformer = TfidfTransformer()
		tfidf = tfidfTransformer.fit_transform(counts)
		if return_model:
			return(countVect, tfidfTransformer)
		else:
			return(tfidf)

	def getTfidf(self, texts, countVect, tfidfTransformer):
		"""Returns the tfidf of new texts from a model which was previously created
		"""
		counts = countVect.transform(texts)
		tfidf = tfidfTransformer.transform(counts)
		return(tfidf)


	def createLDAModel(texts, n_topics, n_passes):
		"""Generates a LDA model from an array of texts
		"""
		tokenizer = RegexpTokenizer(r'\w+')
		#Create EN stop words list
		en_stop = get_stop_words('en')
		#Create p_stemmer of class PorterStemmer
		p_stemmer = PorterStemmer()

		texts_ = []

		# loop through document list
		for i in texts:
		    
		    # clean and tokenize document string
		    raw = i.lower()
		    tokens = tokenizer.tokenize(raw)
		    
		    # remove stop words from tokens
		    stopped_tokens = [i for i in tokens if not i in en_stop]
		    # stem tokens
		    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
		    # add tokens to list
		    texts_.append(stemmed_tokens)

		# turn our tokenized documents into a id <-> term dictionary
		dictionary = corpora.Dictionary(texts_)

		# convert tokenized documents into a document-term matrix
		corpus = [dictionary.doc2bow(text) for text in texts_]

		# generate LDA model
		ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=n_topics, id2word = dictionary, passes=n_passes)

		return(ldamodel)





















