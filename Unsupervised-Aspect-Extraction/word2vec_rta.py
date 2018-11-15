import gensim
import codecs
import numpy as np

class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()


def main(domain):
    source = 'datasets/preprocessed_data/%s/train.txt' % (domain)
    model_file = 'datasets/preprocessed_data/%s/w2v_embedding' % (domain)
    sentences = MySentences(source)
    model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=5, workers=4, iter=10)
    model.save(model_file)
    print(model.most_similar('poverty'))


print ('Pre-training word embeddings ...')
main('mvp')



