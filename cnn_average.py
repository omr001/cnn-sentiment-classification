import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, Chain, iterators, training
from chainer.training import extensions
from gensim.models import word2vec
import json
from collections import defaultdict
import copy
import re
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', dest='gpu', type=int, default=-1)
parser.add_argument('--traindata', dest='traindata', type=str, default='./data/sst5_train_label_sentence.txt')
parser.add_argument('--devdata', dest='devdata', type=str, default='./data/sst5_dev_label_sentence.txt')
parser.add_argument('--testdata', dest='testdata', type=str, default='./data/sst5_test_label_sentence.txt')    
parser.add_argument('--batchsize', dest='batchsize', type=int, default=50)
parser.add_argument('--epoch', dest='epoch', type=int, default=25)
parser.add_argument('classtype', dest='classtype', type=int, default=5)
args = parser.parse_args()

# GPU
if args.gpu != -1:
    from chainer import cuda
    cuda.get_device(args.gpu).use()
    from chainer.cuda import cupy
    xp = cupy
else:
    xp = np

class CNN_average(Chain):
    def __init__(self, vocab_size, embedding_size, input_channel, output_channel_1, output_channel_2, output_channel_3, k1size, k2size, k3size, pooling_units, output_size=args.classtype, train=True):
        super(CNN_average, self).__init__(
            w2e = L.EmbedID(vocab_size, embedding_size),
            conv1 = L.Convolution2D(input_channel, output_channel_1, (k1size, embedding_size)),
            conv2 = L.Convolution2D(input_channel, output_channel_2, (k2size, embedding_size)),
            conv3 = L.Convolution2D(input_channel, output_channel_3, (k3size, embedding_size)),

            l1 = L.Linear(pooling_units, output_size),
        )
        self.output_size = output_size
        self.train = train
        self.embedding_size = embedding_size
        self.ignore_label = 0
        self.w2e.W.data[self.ignore_label] = 0
        self.w2e.W.data[1] = 0  # 非文字
        self.input_channel = input_channel

    def initialize_embeddings(self, word2id):
        #w_vector = word2vec.Word2Vec.load_word2vec_format('./vector/glove.840B.300d.txt', binary=False)  # GloVe
        w_vector = word2vec.Word2Vec.load_word2vec_format('./vector/GoogleNews-vectors-negative300.bin', binary=True)  # word2vec
        for word, id in sorted(word2id.items(), key=lambda x:x[1])[1:]:
            if word in w_vector:
                self.w2e.W.data[id] = w_vector[word]
            else:
                self.w2e.W.data[id] = np.reshape(np.random.uniform(-0.25,0.25,self.embedding_size),(self.embedding_size,))
    
    def __call__(self, x):
        h_list = list()
        ox = copy.copy(x)
        if args.gpu != -1:
            ox.to_gpu()
        
        b = x.shape[0]
        emp_array = xp.array([len(xp.where(x[i][0].data != 0)[0]) for i in range(b)], dtype=xp.float32).reshape(b,1,1,1)
        
        x = xp.array(x.data)
        x = F.tanh(self.w2e(x))
        b, max_len, w = x.shape  # batch_size, max_len, embedding_size
        x = F.reshape(x, (b, self.input_channel, max_len, w))

        c1 = self.conv1(x)
        b, outputC, fixed_len, _ = c1.shape
        tf = self.set_tfs(ox, b, outputC, fixed_len)  # true&flase
        h1 = self.average_pooling(F.relu(c1), b, outputC, fixed_len, tf, emp_array)
        h1 = F.reshape(h1, (b, outputC))
        h_list.append(h1)

        c2 = self.conv2(x)
        b, outputC, fixed_len, _ = c2.shape
        tf = self.set_tfs(ox, b, outputC, fixed_len)  # true&flase
        h2 = self.average_pooling(F.relu(c2), b, outputC, fixed_len, tf, emp_array)
        h2 = F.reshape(h2, (b, outputC))
        h_list.append(h2)

        c3 = self.conv3(x)
        b, outputC, fixed_len, _ = c3.shape
        tf = self.set_tfs(ox, b, outputC, fixed_len)  # true&flase
        h3 = self.average_pooling(F.relu(c3), b, outputC, fixed_len, tf, emp_array)
        h3 = F.reshape(h3, (b, outputC))
        h_list.append(h3)

        h4 = F.concat(h_list)
        y = self.l1(F.dropout(h4, train=self.train))
        return y

    def set_tfs(self, x, b, outputC, fixed_len):
        TF = Variable(x[:,:fixed_len].data != 0, volatile='auto')
        TF = F.reshape(TF, (b, 1, fixed_len, 1))
        TF = F.broadcast_to(TF, (b, outputC, fixed_len, 1))
        return TF

    def average_pooling(self, c, b, outputC, fixed_len, tf, emp_array):
        emp_array = F.broadcast_to(emp_array, (b, outputC, 1, 1))
        masked_c = F.where(tf, c, Variable(xp.zeros((b, outputC, fixed_len, 1)).astype(xp.float32), volatile='auto'))
        sum_c = F.sum(masked_c, axis=2)
        p = F.reshape(sum_c, (b, outputC, 1, 1)) / emp_array
        return p

def create_dataset(fname):
    x_list = list()  # sentence
    y_list = list()  # label
    max_len = 0
    
    for line in open(fname, 'r'):
        spl = line.strip().split(' ', 1)
        y_list.append(spl[0])  # label
        x_list.append(spl[1].split())  # sentence
        max_len = max(len(spl[1].split()), max_len)
    
    # padding
    new_x_list = list()
    for x in x_list:
        pad = ['<EMP>' for i in range(max_len-len(x))]
        new_x_list.append(x+pad)
    
    vec_x_list = list()  # sentence for trainer
    vec_y_list = list()  # label for trainer
    for word_list, y in zip(new_x_list, y_list):
        y = xp.array(int(y), dtype=xp.int32)
        vec_y_list.append(y)
        vec_word_list = list()
        for word in word_list:
            if word == '<EMP>':
                pass
            else:
                #  前処理
                word = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", word)
                word = re.sub(r"\s{2,}", " ", word)
                word = word.strip().lower()

            vec_word_list.append(word2id[word])

        vec_word_list = xp.array(vec_word_list, dtype=xp.int32)
        vec_x_list.append(vec_word_list)

    return vec_x_list, vec_y_list 
    

if __name__ == "__main__":
    word2id = defaultdict(lambda: len(word2id))
    word2id['<EMP>']
    word2id['']  # 前処理後の記号

    x_train, y_train = create_dataset(args.traindata)
    x_dev, y_dev = create_dataset(args.devdata)
    x_test, y_test = create_dataset(args.testdata)
    
    print('finish data loding ...')
    
    train = [(x, y) for x, y in zip(x_train, y_train)]
    dev = [(x, y) for x, y in zip(x_dev, y_dev)]
    test = [(x, y) for x, y in zip(x_test, y_test)]
    
    vocab_size = len(word2id)
    emb_size = 300
    ic = 1
    oc1 = 100
    oc2 = 100
    oc3 = 100
    k1 = 3
    k2 = 4
    k3 = 5
    p_units = 300
    cnn = CNN_average(vocab_size, emb_size, ic, oc1, oc2, oc3, k1, k2, k3, p_units)
    cnn.initialize_embeddings(word2id) 
    model = L.Classifier(cnn)
    # gpu
    if args.gpu != -1:
        model.to_gpu()
    
    optimizer = optimizers.AdaDelta()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(3))

    log_dir = 'result_SST'+str(args.classtype)
    log_name = 'CNN_average_pooling_SST'+str(args.classtype)
    
    batch_size = args.batch_size
    train_iter = iterators.SerialIterator(train, batch_size=batch_size)
    dev_iter = iterators.SerialIterator(dev, batch_size=batch_size, repeat=False, shuffle=False)
    test_iter = iterators.SerialIterator(test, batch_size=batch_size, repeat=False, shuffle=False)
    
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=log_dir)

    eval_model = model.copy()
    eval_cnn = eval_model.predictor
    eval_cnn.train = False

    dev_evaluator = extensions.Evaluator(dev_iter, eval_model)
    dev_evaluator.name = 'dev'
    trainer.extend(dev_evaluator)
    test_evaluator = extensions.Evaluator(test_iter, eval_model)
    test_evaluator.name = 'test'
    trainer.extend(test_evaluator)
    trainer.extend(extensions.LogReport(log_name=log_name))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/accuracy', 'main/loss', 'test/main/accuracy', 'test/main/loss', 'dev/main/accuracy',
         'dev/main/loss']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()
    print(trainer.observation.keys())
