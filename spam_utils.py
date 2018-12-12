import csv
import os
import string

import numpy as np
import torch
from nltk import word_tokenize, SnowballStemmer
from nltk.corpus import stopwords
from progressbar import progressbar
from torch import nn
from torch.autograd import Variable
from torch.nn import Module
from torch.utils.data import Dataset

from base_utils import History, AverageMeter, save_model


def read_csv(filename):
    with open(filename) as csvDataFile:
        return np.array(list(csv.reader(csvDataFile)))


def pre_process(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    words = ""
    for i in text:
        stemmer = SnowballStemmer("english")
        words += (stemmer.stem(i)) + " "
    return words


class Corpus:
    def __init__(self):
        super().__init__()
        self.word2idx = {'<pad>': 0}
        self.idx2word = ['<pad>']
        self.sentence_lengths = []
        traindata = read_csv('data/train.csv')
        traindata = traindata[1:]
        test_data = read_csv('data/test.csv')
        test_data = test_data[1:]
        traindata = np.array([[row[0], ','.join(row[1:]).lower()] for row in traindata])
        test_data = np.array([[row[0],','.join(row[1:]).lower()] for row in test_data])
        labels = np.array(list(map(lambda x: 0.0 if x == 'ham' else 1.0, traindata[:, 0])))
        self.test_smsids = np.array(list(map(int, test_data[:, 0])),dtype=int)
        raw_sentences = traindata[:, 1]
        self.train_sentences = []
        self.test_sentences = []
        print('reading and preprocessing training data')
        for sentence in progressbar(raw_sentences):
            sentence = pre_process(sentence)
            self.train_sentences.append(sentence)
            words = sentence.split()
            self.sentence_lengths.append(len(words))
            for word in words:
                if word not in self.word2idx.keys():
                    self.word2idx[word] = len(self.idx2word)
                    self.idx2word.append(word)
        for sentence in progressbar(test_data[:,1]):
            sentence = pre_process(sentence)
            self.test_sentences.append(sentence)
        self.word2idx['<unknown>'] = len(self.idx2word)
        self.idx2word.append('<unknown>')
        self.labels = labels

    def __len__(self):
        return len(self.idx2word)


class Model(Module):
    def __init__(self, corpus: Corpus, num_embeddings=None, embedding_dim=50, hidden_size=64, hidden_dim=32):
        super().__init__()
        num_embeddings = len(corpus) if num_embeddings is None else num_embeddings
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.fcs = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=hidden_dim),
                                 nn.ReLU(),
                                 nn.Dropout(),
                                 nn.Linear(in_features=hidden_dim, out_features=1),
                                 nn.Sigmoid()
                                 )

    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.lstm(embed)
        return self.fcs(out[:, -1, :])


class SpamSet(Dataset):
    def __init__(self, train, corpus: Corpus, max_len):
        super().__init__()
        self.train = train
        self.corpus = corpus
        self.num_total = len(corpus.labels)
        self.num_train = int(0.8 * self.num_total)
        self.num_dev = self.num_total - self.num_train
        self.max_len = max_len

    def __getitem__(self, index):
        idx = index if self.train else self.num_train + index
        x = self.sentence_to_indices(self.corpus.train_sentences[idx], self.max_len)
        y = self.corpus.labels[idx]
        return torch.LongTensor(x), torch.FloatTensor([y])

    def __len__(self):
        return self.num_train if self.train else self.num_dev

    def sentence_to_indices(self, sentence, max_len):
        words = sentence.split()
        result = np.zeros(max_len, dtype=np.long)
        for i, word in enumerate(words):
            if i >= max_len: break
            result[i] = self.corpus.word2idx[word]
        return result


def fit(model, loss_fn, optimizer, dataloaders, metrics_functions=None, num_epochs=1, scheduler=None, begin_epoch=0,
        save=True,
        save_model_dir='data/models', history=None, use_progressbar=False, plot_every_epoch=False):
    if metrics_functions is None:
        metrics_functions = {}
    if save and save_model_dir is None:
        raise Exception('save_model is True but no directory is specified.')
    if save:
        os.system('mkdir -p ' + save_model_dir)
    num_epochs += begin_epoch
    if history is None:
        history = History(['loss', *metrics_functions.keys()])
    max_len = dataloaders['train'].dataset.max_len
    for epoch in range(begin_epoch, num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        for phase in ['train', 'dev']:
            meters = {'loss': AverageMeter()}
            for k in metrics_functions.keys():
                meters[k] = AverageMeter()
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train()
            else:
                model.eval()
            loaders = progressbar(dataloaders[phase]) if use_progressbar else dataloaders[phase]
            for data in loaders:
                x, y = data
                x = x.reshape((-1, max_len))
                nsamples = x.shape[0]
                x_var = Variable(x.cuda())
                y_var = Variable(y.cuda())
                optimizer.zero_grad()
                scores = model(x_var)
                loss = loss_fn(scores, y_var)

                meters['loss'].update(loss.item(), nsamples)
                for k, f in metrics_functions.items():
                    result = f(scores.detach().cpu().numpy(),
                               y.detach().cpu().numpy().astype(np.int64))
                    meters[k].update(result, nsamples)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            s = 'Epoch {}/{}, {}, loss = {:.4f}'.format(epoch + 1, num_epochs, phase, meters['loss'].avg)

            for k in metrics_functions.keys():
                s += ', {} = {:.4f}'.format(k, meters[k].avg)
            print(s)
            history.records['loss'][phase].append(meters['loss'].avg)
            for k in metrics_functions.keys():
                history.records[k][phase].append(meters[k].avg)
        if save:
            save_model(model, optimizer, epoch, save_model_dir)
        if plot_every_epoch:
            history.plot()
    if not plot_every_epoch:
        history.plot()


def read_glove_vecs(glove_file='data/glove.6B.50d.txt'):
    with open(glove_file, 'r', encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


if __name__ == '__main__':
    # print(pre_process('Ok lar... Joking wif u oni...'))
    corpus = Corpus()
    print(len(corpus))
