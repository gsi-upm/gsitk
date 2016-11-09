import re

__author__ = 'NachoCP'


class TwitterTokenizer:

    def __init__(self):
        self.eyes = r"[8:=;]"
        self.nose = r"['`\-]?"
        self.FLAGS = re.MULTILINE | re.DOTALL

    def hashtag(self, text):
        text = text.group()
        hashtag_body = text[1:]
        if hashtag_body.isupper():
            result = "<hashtag> {} <allcaps>".format(hashtag_body)
        else:
            result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=self.FLAGS))
        return result

    @staticmethod
    def allcaps(text):
        text = text.group()
        return text.lower() + " <allcaps>"

    def tokenize(self, text):
        text = self.re_sub(text, r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
        text = self.re_sub(text, r"/", " / ")
        text = self.re_sub(text, r"@\w+", "<user>")
        text = self.re_sub(text, r"{}{}[)dD]+|[)dD]+{}{}".format(self.eyes, self.nose, self.nose, self.eyes), "<smile>")
        text = self.re_sub(text, r"{}{}p+".format(self.eyes, self.nose), "<lolface>")
        text = self.re_sub(text, r"{}{}\(+|\)+{}{}".format(self.eyes, self.nose, self.nose, self.eyes), "<sadface>")
        text = self.re_sub(text, r"{}{}[\/|l*]".format(self.eyes, self.nose), "<neutralface>")
        text = self.re_sub(text, r"<3", "<heart>")
        text = self.re_sub(text, r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
        text = self.re_sub(text, r"#\S+", 'hashtag')
        text = self.re_sub(text, r"([!?.]){2,}", r"\1 <repeat>")
        text = self.re_sub(text, r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

        # I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
        text = self.re_sub(text, r"([A-Z]){2,}", 'allcaps')
        return text.lower()

    def re_sub(self, input_string, pattern, repl):
        return re.sub(pattern, repl, input_string, flags=self.FLAGS)

    @staticmethod
    def twitter_features(text):
        length = float(len(text.split()))
        features = list()

        mention_count = ('@COUNT', float(len(re.findall('@username', text))) / length)
        hashtag_count = ('#COUNT', float(len(re.findall('#', text))) / length)
        rt_count = ('RT', float(len(re.findall('RT @username', text))) / length)
        url_count = ('URL', float(len(re.findall('http[s]?://', text))) / length)
        pic_count = ('PIC', float(len(re.findall('pic.twitter.com', text))) / length)

        features.append(mention_count)
        features.append(hashtag_count)
        features.append(rt_count)
        features.append(url_count)
        features.append(pic_count)
        return features
