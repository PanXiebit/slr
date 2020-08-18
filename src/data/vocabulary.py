

class Vocabulary():
    def __init__(self, vocab_file):
        PAD_token = 0
        self.vocab_file = vocab_file
        self.word2index = {'<PAD>': PAD_token}
        self.index2word = {PAD_token: '<PAD>'}
        self.num_words = 1

        count = 0
        with open(self.vocab_file, 'r') as fid:
            for line in fid:
                if count != 0:
                    line = line.strip().split(' ')
                    assert len(line) == 2, "The format of line should be '(word, num)' in vocabulary file "
                    word = line[0]
                    if word not in self.word2index:
                        self.word2index[word] = self.num_words
                        self.index2word[self.num_words] = word
                        self.num_words += 1
                count += 1
        UNK_token = self.num_words
        BOS_token = self.num_words + 1
        EOS_token = self.num_words + 2
        BLANK_token = self.num_words + 3
        self.word2index['<UNK>'] = UNK_token
        self.word2index['<BOS>'] = BOS_token
        self.word2index['<EOS>'] = EOS_token
        self.word2index['<BLANK>'] = BLANK_token
        self.index2word[UNK_token] = '<UNK>'
        self.index2word[BOS_token] = '<BOS>'
        self.index2word[EOS_token] = '<EOS>'
        self.index2word[BLANK_token] = '<BLANK>'
        self.num_words += 4

    def bos(self):
        return self.word2index["<BOS>"]

    def eos(self):
        return self.word2index['<EOS>']

    def blank(self):
        return self.word2index['<BLANK>']

    def unk(self):
        return self.word2index['<UNK>']

    def pad(self):
        return self.word2index["<PAD>"]

    def vocab_size(self):
        return self.num_words

if __name__ == "__main__":
    Vocab = Vocabulary(vocab_file="/root/workspace/SLR-SLT/Data/slr-phoenix14/newtrainingClasses.txt")
    print(Vocab.word2index.keys())