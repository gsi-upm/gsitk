import pandas,csv,re

class Utilities:



    def find_ngrams(input_list, n):
        """Combines the different input_list to create a set of Ngrams.
           Params
           -input_list, list of string that it is wanted to combine
           -n, number of times that it is wanted to combine, e.g: for bigrams 2, for trigrams 3.
        """
        return zip(*[input_list[i:] for i in range(n)])

    def merge_two_dicts(x, y):
        """Given two dicts, merge them into a new dict as a shallow copy.
        """
        z = x.copy()
        z.update(y)
        return z

    def parse_dict_to_tuples(dictionary):
        """Converts the key value pairs of a dictionary into tuples
        """
        tuples = [(key,dictionary[key]) for key in dictionary]
        return tuples

    def load_EmoLex(path,lang):
        """Loading different dictionaries
        """
        emoLex_mapping = {}
        emoLex = pandas.read_excel(path+"NRC-Emotion-Lexicon-v0.92-InManyLanguages-web.xlsx")

        emoLex = emoLex.to_dict('records')

        for element in emoLex:
            emotion_list = [key for key,value in element.iteritems() if value != 0 and type(value) is int]
            if len(emotion_list) == 0 or type(element[lang]) is int:
                continue

            emotion_list = [emotion+'_emoLex' for emotion in emotion_list]
            emoLex_mapping[element[lang]] = emotion_list

        return emoLex_mapping

    def load_MoralDIC(path):
        with open(path+"moral-foundations-categories.txt",'r') as tsv:
            categories_LIWC = [line.strip().split('\t') for line in tsv]
        with open(path+"moral foundations dictionary.dic",'r') as tsv:
            words_MoralDIC = [line.strip().split('\t') for line in tsv]
        categories_MoralDIC = {key: value for (key, value) in categories_LIWC}

        dictionary_MoralDIC = {}

        for word_categories in words_MoralDIC:
            word = re.sub('[*]','',word_categories[0])
            for category in word_categories[1:]:
                if category != '':
                    moral_categories = category.split(' ')
            categories = [categories_MoralDIC[str(category).strip()]+'_moral' for category in moral_categories]
            dictionary_MoralDIC[word] = categories

        return dictionary_MoralDIC

    def load_NRC_hashtag_emotion(path):
        hashtag_emotion_lexicon = {}
        with open(path+'NRC-Hashtag-Emotion-Lexicon-v0.2.txt','rb') as tsv:
            tsvin = csv.reader(tsv,delimiter='\t')
            for row in tsvin:
                if row[1] in hashtag_emotion_lexicon:
                    hashtag_emotion_lexicon[row[1]][row[0]] = row[2]
                else:
                    hashtag_emotion_lexicon[row[1]] = {}
                    hashtag_emotion_lexicon[row[1]][row[0]] = row[2]
        return hashtag_emotion_lexicon

    def load_MRC(path):
        dictionary_MRC = {}
        with open(path+"MRC_dictionary.txt") as mrc:
            lines = mrc.readlines()

            for line in lines:
                p = line.split()
                dictionary_MRC[p[0].lower()] = {}
                dictionary_MRC[p[0].lower()]['I'] = float(p[1])
                dictionary_MRC[p[0].lower()]['AOA'] = float(p[3])
                dictionary_MRC[p[0].lower()]['F'] = float(p[5])
                dictionary_MRC[p[0].lower()]['C'] = float(p[7])

        return dictionary_MRC