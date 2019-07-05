

class Tokens:
    
    def __init__(self):
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2


tokens = Tokens()
# Vocabulary
class Voc:
    def __init__(self, name, tokens = tokens):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {tokens.PAD_token: 'PAD', tokens.SOS_token: 'SOS', tokens.EOS_token: 'EOS'}
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return 
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(len(keep_words), len(self.word2index),
                                                    len(keep_words)/len(self.word2index)))
        
        self.word2index = {}
        self.word2count = {}
        self.index2word = {tokens.PAD_token: 'PAD', tokens.SOS_token: 'SOS', tokens.EOS_token: 'EOS'}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)




