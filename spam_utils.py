import csv
import os
import string

import numpy as np
import torch
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from progressbar import progressbar
from sklearn.metrics import precision_score
from torch import nn
from torch.autograd import Variable
from torch.nn import Module
from torch.utils.data import Dataset

from base_utils import History, AverageMeter, save_model, read_csv, load_model
import visdom


def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        print('reading pretrained vectors..')
        for line in progressbar(list(f)):
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
        print('done.')
    return words_to_index, index_to_words, word_to_vec_map


def pre_process(text: str):
    # delete punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    # delete stop words and split them
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    words = ""
    for i in text:
        stemmer = SnowballStemmer("english")
        # lower and return word stem
        words += (stemmer.stem(i)) + " "
    return words


class Corpus_v3:
    def __init__(self, glove_file):
        super().__init__()
        words_to_index, index_to_words, word_to_vec_map = read_glove_vecs(glove_file)
        self.embedding_dim = word_to_vec_map['cucumber'].shape[0]
        self.weights = np.zeros((len(index_to_words) + 1, self.embedding_dim))
        for word, idx in words_to_index.items():
            self.weights[idx] = word_to_vec_map[word]
        words_to_index['<pad>'] = 0
        index_to_words[0] = '<pad>'
        # index_to_words.append('<unknown>')
        # words_to_index['<unknown>'] = len(index_to_words) - 1
        self.word2idx = words_to_index
        self.idx2word = index_to_words
        os.system('mkdir -p data/preprocess')
        sentence_paths = 'data/preprocess/{}'
        train_preprocess_sentence_path = sentence_paths.format('train.npy')
        test_preprocess_sentence_path = sentence_paths.format('test.npy')
        traindata = read_csv('data/train.csv')
        traindata = traindata[1:]
        test_data = read_csv('data/test.csv')
        test_data = test_data[1:]
        traindata = np.array([[row[0], ','.join(row[1:]).lower()] for row in traindata])
        test_data = np.array([[row[0], ','.join(row[1:]).lower()] for row in test_data])
        labels = np.array(list(map(lambda x: 0.0 if x == 'ham' else 1.0, traindata[:, 0])))
        self.test_smsids = np.array(list(map(int, test_data[:, 0])), dtype=int)
        raw_sentences = traindata[:, 1]
        self.train_sentences: list = []
        self.test_sentences = []
        if os.path.exists(train_preprocess_sentence_path):
            self.train_sentences = np.load(train_preprocess_sentence_path).tolist()
        else:
            for sentence in progressbar(raw_sentences):
                sentence = pre_process(sentence)
                self.train_sentences.append(sentence)
            np.save(train_preprocess_sentence_path, self.train_sentences)
        if os.path.exists(test_preprocess_sentence_path):
            self.test_sentences = np.load(test_preprocess_sentence_path).tolist()
        else:
            for sentence in progressbar(test_data[:, 1]):
                sentence = pre_process(sentence)
                self.test_sentences.append(sentence)
            np.save(test_preprocess_sentence_path, self.test_sentences)

        self.labels = labels

    def __len__(self):
        return len(self.idx2word)


class Model_v3(Module):
    def __init__(self, corpus: Corpus_v3):
        super().__init__()
        num_embeddings = len(corpus)
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=corpus.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(corpus.weights))
        for param in self.embedding.parameters():
            param.requires_grad = False
        self.lstm = nn.LSTM(input_size=corpus.embedding_dim, hidden_size=512, batch_first=True)
        self.fcs = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        embed = self.embedding(x)
        out, _ = self.lstm(embed)
        return self.fcs(out[:, -1, :])


class SpamSet_v3(Dataset):
    def __init__(self, train, corpus: Corpus_v3, max_len):
        super().__init__()
        self.train = train
        self.corpus = corpus
        self.num_total = len(corpus.labels)
        self.num_train = int(0.8 * self.num_total)
        self.num_dev = self.num_total - self.num_train
        self.max_len = max_len
        self.perm = range(self.num_total)

    def __getitem__(self, index):
        idx = index if self.train else self.num_train + index
        idx = self.perm[idx]
        sentence = self.corpus.train_sentences[idx]
        x = self.sentence_to_indices(sentence, self.max_len)
        y = self.corpus.labels[idx]
        return torch.LongTensor(x), torch.FloatTensor([y])

    def __len__(self):
        return self.num_train if self.train else self.num_dev

    def set_perm(self, perm):
        self.perm = perm

    def sentence_to_indices(self, sentence, max_len):
        words = sentence.split()
        result = np.zeros(max_len, dtype=np.long)
        for i, word in enumerate(words):
            if i >= max_len: break
            if word in self.corpus.word2idx.keys():
                result[i] = self.corpus.word2idx[word]
        return result


viz = visdom.Visdom()


def create_vis_plot(_xlabel, _ylabel, _title, _legend, num_keys):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, num_keys)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, window1, update_type, *args):
    viz.line(
        X=torch.ones((1, len(args))).cpu() * iteration,
        Y=torch.Tensor(args).unsqueeze(0).cpu(),
        win=window1,
        update=update_type
    )


def fit_v3(model, loss_fn, optimizer, dataloaders, metrics_functions=None, num_epochs=1, scheduler=None, begin_epoch=0,
           save_model_dir='data/models', history=None, use_progressbar=False):
    if metrics_functions is None:
        metrics_functions = {}

    os.system('mkdir -p ' + save_model_dir)
    num_epochs += begin_epoch
    if history is None:
        history = History(['loss', *metrics_functions.keys()])
    max_len = dataloaders['train'].dataset.max_len
    num_total = dataloaders['train'].dataset.num_total
    vis_title = 'spam messgae classification'
    vis_legend = ['loss', *metrics_functions.keys()]
    epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend, len(metrics_functions) + 1)
    for epoch in range(begin_epoch, num_epochs):
        perm = np.random.permutation(num_total)
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        for phase in ['train', 'dev']:
            if (epoch - begin_epoch) / (num_epochs - begin_epoch) > 0.5:
                dataloaders[phase].dataset.set_perm(perm)
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
            for it, data in enumerate(loaders):
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
                    result = f(y.detach().cpu().numpy().astype(np.int64),
                               scores.detach().cpu().numpy())
                    meters[k].update(result, nsamples)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            s = 'Epoch {}/{}, {}, loss = {:.4f}'.format(epoch + 1, num_epochs, phase, meters['loss'].avg)
            avgs = []
            for k in metrics_functions.keys():
                s += ', {} = {:.4f}'.format(k, meters[k].avg)
                avgs.append(meters[k].avg)
            print(s)
            history.records['loss'][phase].append(meters['loss'].avg)
            update_vis_plot(epoch, epoch_plot,
                            'append', meters['loss'].avg, *avgs)
            for k in metrics_functions.keys():
                history.records[k][phase].append(meters[k].avg)
        save_model(model, optimizer, epoch, save_model_dir)


def train_v3(model, loss_fn, optimizer, dataloaders, scheduler, num_epochs):
    def compute_precision(y_true, y_pred):
        return precision_score(y_true.squeeze(), y_pred.squeeze() >= 0.5)

    metrics = {
        'precision': compute_precision
    }
    fit_v3(model, loss_fn, optimizer, dataloaders, scheduler=scheduler, metrics_functions=metrics,
           num_epochs=num_epochs)


def test_v3(corpus: Corpus_v3, model, optimizer, max_len):
    load_model(model, optimizer, 'data/model')
    model.eval()

    sentences = corpus.test_sentences
    smsids = corpus.test_smsids
    num_test = len(sentences)
    sentence_indices = np.zeros((num_test, max_len), dtype=np.long)
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        for j, word in enumerate(words):
            if j >= max_len: break
            if word in corpus.word2idx.keys():
                sentence_indices[i, j] = corpus.word2idx[word]

    predictions = [['SmsId', 'Label']]
    for i in progressbar(range(sentence_indices.shape[0])):
        out = model(torch.LongTensor(sentence_indices[i]).reshape((1, -1)).cuda())
        pred = out >= 0.5
        pred = pred.cpu().numpy().squeeze()
        predictions.append([smsids[i], 'spam' if pred == 1 else 'ham'])
    with open('data/submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(predictions)
