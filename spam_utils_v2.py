import csv
import os
import string

import numpy as np
import torch
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from progressbar import progressbar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score
from torch import nn
from torch.autograd import Variable
from torch.nn import Module
from torch.utils.data import Dataset

from base_utils import History, AverageMeter, save_model, read_csv, load_model


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


class Corpus_v2:
    def __init__(self):
        super().__init__()
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
        self.train_sentences = []
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
        vectorizer = TfidfVectorizer("english")
        textFeatures = self.train_sentences.copy() + self.test_sentences.copy()
        features_total = vectorizer.fit_transform(textFeatures)
        num_train_and_dev = len(self.train_sentences)
        self.train_features = features_total[0:num_train_and_dev]
        self.test_features = features_total[num_train_and_dev:]
        self.labels = labels


class Model_v2(Module):
    def __init__(self, corpus: Corpus_v2):
        super().__init__()
        feature_size = corpus.train_features[0].toarray().squeeze().shape[0]
        self.fcs = nn.Sequential(nn.Linear(in_features=feature_size, out_features=128),
                                 nn.ReLU(),
                                 nn.Linear(in_features=128, out_features=64),
                                 nn.ReLU(),
                                 nn.Linear(in_features=64, out_features=1),
                                 nn.Sigmoid()
                                 )

    def forward(self, x):
        return self.fcs(x)


class SpamSet_v2(Dataset):
    def __init__(self, train, corpus: Corpus_v2):
        super().__init__()
        self.train = train
        self.corpus = corpus
        self.num_total = len(corpus.labels)
        self.num_train = int(0.7 * self.num_total)
        self.num_dev = self.num_total - self.num_train

    def __getitem__(self, index):
        idx = index if self.train else self.num_train + index
        x = self.corpus.train_features[idx]
        y = self.corpus.labels[idx]
        return torch.FloatTensor(x.toarray()), torch.FloatTensor([y])

    def __len__(self):
        return self.num_train if self.train else self.num_dev


def fit_v2(model, loss_fn, optimizer, dataloaders, metrics_functions=None, num_epochs=1, scheduler=None, begin_epoch=0,
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


def train_v2(model, loss_fn, optimizer, dataloaders, scheduler, num_epochs):
    def compute_precision(y_true, y_pred):
        return precision_score(y_true.squeeze(), y_pred.squeeze() >= 0.5)

    metrics = {
        'precision': compute_precision
    }
    fit_v2(model, loss_fn, optimizer, dataloaders, scheduler=scheduler, metrics_functions=metrics,
           num_epochs=num_epochs)


def test_v2(corpus: Corpus_v2, model, optimizer):
    load_model(model, optimizer, 'data/model')
    model.eval()
    test_features = corpus.test_features.toarray()
    smsids = corpus.test_smsids

    predictions = [['SmsId', 'Label']]
    for i in progressbar(range(test_features.shape[0])):
        out = model(torch.FloatTensor(test_features[i]).reshape((1, -1)).cuda())
        pred = out >= 0.5
        pred = pred.cpu().numpy().squeeze()
        predictions.append([smsids[i], 'spam' if pred == 1 else 'ham'])
    with open('data/submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(predictions)


if __name__ == '__main__':
    corpus = Corpus_v2()
    s = SpamSet_v2(True, corpus)
    x, y = s[1]
    print(x.shape)
