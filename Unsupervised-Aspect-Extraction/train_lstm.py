import argparse
import logging
import numpy as np
from time import time
import utils as U
import codecs

from keras.preprocessing import sequence
import reader as dataset

from optimizers import get_optimizer

from model import create_model, create_lstm_model
import keras.backend as K

from keras.models import load_model
from tqdm import tqdm

logging.basicConfig(
                    #filename='out.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', default="output", help="The path to the output directory")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=200, help="Embeddings dimension (default=200)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=50, help="Batch size (default=50)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=9000, help="Vocab size. '0' means no limit (default=9000)")
parser.add_argument("-as", "--aspect-size", dest="aspect_size", type=int, metavar='<int>', default=8, help="The number of aspects specified by users (default=14)")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', default="datasets/preprocessed_data/mvp/w2v_embedding", help="The path to the word embeddings file")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=100, help="Number of epochs (default=15)")
parser.add_argument("-n", "--neg-size", dest="neg_size", type=int, metavar='<int>', default=20, help="Number of negative instances (default=20)")
parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=42, help="Random seed (default=42)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='mvp', help="domain of the corpus {restaurant, beer}")
parser.add_argument("--ortho-reg", dest="ortho_reg", type=float, metavar='<float>', default=0.1, help="The weight of orthogonol regularizaiton (default=0.1)")

args = parser.parse_args()
out_dir = args.out_dir_path + '/' + args.domain
U.mkdir_p(out_dir)
U.print_args(args)

assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}
assert args.domain in {'restaurant', 'beer', 'mvp', 'space'}

if args.seed > 0:
    np.random.seed(args.seed)


# ###############################################################################################################################
# ## Prepare data
# #



vocab, train_x, test_x, overall_maxlen = dataset.get_data(args.domain, vocab_size=args.vocab_size, maxlen=args.maxlen)
train_x = sequence.pad_sequences(train_x, maxlen=overall_maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=overall_maxlen)

print('Number of training examples: ', len(train_x))
print('Length of vocab: ', len(vocab))

def sentence_batch_generator(data, batch_size):
    n_batch = int(len(data) / batch_size)
    batch_count = 0
    np.random.shuffle(data)

    while True:
        if batch_count == n_batch:
            np.random.shuffle(data)
            batch_count = 0

        batch = data[batch_count*batch_size: (batch_count+1)*batch_size]
        batch_count += 1
        yield batch

def negative_batch_generator(data, batch_size, neg_size):
    data_len = data.shape[0]
    dim = data.shape[1]

    while True:
        indices = np.random.choice(data_len, batch_size * neg_size)
        samples = data[indices].reshape(batch_size, neg_size, dim)
        yield samples



###############################################################################################################################
## Optimizaer algorithm
#



optimizer = get_optimizer(args)



###############################################################################################################################
## Building model



logger.info('  Building model')


def max_margin_loss(y_true, y_pred):
    return K.mean(y_pred)

model = create_lstm_model(args, overall_maxlen, vocab)
# freeze the word embedding layer
model.get_layer('word_emb').trainable=False
model.summary()
model.compile(optimizer=optimizer, loss=max_margin_loss, metrics=[max_margin_loss])
if False:
    model.load_weights(out_dir + '/steps_lstm_model_param_0_' + str(args.aspect_size))
    vector_output = K.function([model.get_layer("sentence_input").input, model.get_layer("neg_input").input],
                               [model.get_layer("max_margin").input[0], model.get_layer("max_margin").input[1],
                                model.get_layer("max_margin").input[2],
                                K.l2_normalize(model.get_layer("max_margin").input[0], axis=-1),
                                K.l2_normalize(model.get_layer("max_margin").input[1], axis=-1),
                                K.l2_normalize(model.get_layer("max_margin").input[2], axis=-1),
                                ])
    sen_gen = sentence_batch_generator(train_x, args.batch_size)
    neg_gen = negative_batch_generator(train_x, args.batch_size, args.neg_size)

    outputs = vector_output([next(sen_gen), next(neg_gen)])
    z_s = outputs[3]
    z_n = outputs[4]
    r_s = outputs[5]

    steps = 20

    pos = np.sum(z_s * r_s, axis=-1, keepdims=True)
    pos = np.repeat(pos, steps, axis=1)
    # colin "dim" to "axis"
    r_s = np.expand_dims(r_s, axis=-2)
    r_s = np.repeat(r_s, steps, axis=1)
    neg = np.sum(z_n * r_s, axis=-1)
    loss = np.sum(np.maximum(0., (1. - pos + neg)), axis=-1, keepdims=True)
    exit()


###############################################################################################################################
## Training
#


logger.info('--------------------------------------------------------------------------------------------------------------------------')

vocab_inv = {}
for w, ind in vocab.items():
    vocab_inv[ind] = w


sen_gen = sentence_batch_generator(train_x, args.batch_size)
neg_gen = negative_batch_generator(train_x, args.batch_size, args.neg_size)
# batches_per_epoch = len(train_x) / args.batch_size
batches_per_epoch = 1000

min_loss = float('inf')
for ii in range(args.epochs):
    t0 = time()
    loss, max_margin_loss = 0., 0.

    for b in tqdm(range(batches_per_epoch)):
        sen_input = next(sen_gen)
        neg_input = next(neg_gen)

        batch_loss, batch_max_margin_loss = model.train_on_batch([sen_input, neg_input], np.ones((args.batch_size, 1)))
        loss += batch_loss / batches_per_epoch
        max_margin_loss += batch_max_margin_loss / batches_per_epoch

    tr_time = time() - t0


    model.save_weights(out_dir + '/steps_lstm_model_param_' + str(ii) + '_' + str(args.aspect_size))

    if loss < min_loss:
        min_loss = loss
        word_emb = K.get_value(model.get_layer('word_emb').embeddings)
        aspect_emb = K.get_value(model.get_layer('aspect_emb').W)
        word_emb = word_emb / np.linalg.norm(word_emb, axis=-1, keepdims=True)
        aspect_emb = aspect_emb / np.linalg.norm(aspect_emb, axis=-1, keepdims=True)
        aspect_file = codecs.open(out_dir+'/lstm_aspect_' + str(args.aspect_size) + '.log', 'w', 'utf-8')
        model.save_weights(out_dir+'/lstm_model_param_' + str(args.aspect_size))

        for ind in range(len(aspect_emb)):
            desc = aspect_emb[ind]
            sims = word_emb.dot(desc.T)
            ordered_words = np.argsort(sims)[::-1]
            desc_list = [vocab_inv[w] + ":" + str(sims[w]) for w in ordered_words[:100]]
            print('Aspect %d:' % ind)
            print(desc_list)
            aspect_file.write('Aspect %d:\n' % ind)
            aspect_file.write(' '.join(desc_list) + '\n\n')

    logger.info('Epoch %d, train: %is' % (ii, tr_time))
    logger.info('Total loss: %.4f, max_margin_loss: %.4f, ortho_reg: %.4f' % (loss, max_margin_loss, loss-max_margin_loss))
            
    







