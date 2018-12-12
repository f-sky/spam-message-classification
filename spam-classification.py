from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.metrics import f1_score, precision_score
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from base_utils import load_model
from spam_utils import *
from sklearn.metrics import accuracy_score

from spam_utils_v2 import *

num_epochs = 200
max_len = 30


def train(model, loss_fn, optimizer, dataloaders):
    def compute_acc(x, y):
        return accuracy_score(y.squeeze(), x.squeeze() > 0.5)

    def compute_f1(x, y):
        return f1_score(y.squeeze(), x.squeeze() > 0.5)

    def compute_precision(x, y):
        return precision_score(y.squeeze(), x.squeeze() > 0.5)

    metrics = {
        # 'accuracy': compute_acc,
        'precision': compute_precision,
        'f1': compute_f1,
    }
    fit(model, loss_fn, optimizer, dataloaders, metrics_functions=metrics, num_epochs=num_epochs)


def test(corpus: Corpus, model, optimizer):
    load_model(model, optimizer, 'data/model')
    model.eval()
    data = read_csv('data/test.csv')
    data = data[1:]
    data = np.array([[row[0], ','.join(row[1:]).lower()] for row in data])
    smsids = np.array(list(map(int, data[:, 0])))
    sentences = data[:, 1]
    num_test = len(sentences)
    sentence_indices = np.zeros((num_test, max_len), dtype=np.long)
    for i, sentence in enumerate(sentences):
        words = word_tokenize(sentence)
        for j, word in enumerate(words):
            if j >= max_len: break
            if word in corpus.word2idx.keys():
                sentence_indices[i, j] = corpus.word2idx[word]
            else:
                sentence_indices[i, j] = corpus.word2idx['<unknown>']
    predictions = [['SmsId', 'Label']]
    for i in progressbar(range(sentence_indices.shape[0])):
        out = model(torch.LongTensor(sentence_indices[i]).reshape((1, -1)).cuda())
        pred = out > 0.5
        pred = pred.cpu().numpy().squeeze()
        predictions.append([smsids[i], 'spam' if pred == 1 else 'ham'])
    with open('data/submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(predictions)


def train_v2(model, loss_fn, optimizer, dataloaders):
    def compute_precision(x, y):
        return precision_score(y.squeeze(), x.squeeze() > 0.5)

    metrics = {
        'precision': compute_precision
    }
    fit_v2(model, loss_fn, optimizer, dataloaders, metrics_functions=metrics, num_epochs=num_epochs)


def test_v2(corpus: Corpus_v2, model, optimizer):
    load_model(model, optimizer, 'data/model')
    model.eval()
    data = read_csv('data/test.csv')
    data = data[1:]
    data = np.array([[row[0], ','.join(row[1:]).lower()] for row in data])
    smsids = np.array(list(map(int, data[:, 0])))
    sentences = data[:, 1]
    num_test = len(sentences)
    sentence_indices = np.zeros((num_test, max_len), dtype=np.long)
    for i, sentence in enumerate(sentences):
        words = word_tokenize(sentence)
        for j, word in enumerate(words):
            if j >= max_len: break
            if word in corpus.word2idx.keys():
                sentence_indices[i, j] = corpus.word2idx[word]
            else:
                sentence_indices[i, j] = corpus.word2idx['<unknown>']
    predictions = [['SmsId', 'Label']]
    for i in progressbar(range(sentence_indices.shape[0])):
        out = model(torch.LongTensor(sentence_indices[i]).reshape((1, -1)).cuda())
        pred = out > 0.5
        pred = pred.cpu().numpy().squeeze()
        predictions.append([smsids[i], 'spam' if pred == 1 else 'ham'])
    with open('data/submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(predictions)


def train_classifier(corpus: Corpus, mode):
    textFeatures = corpus.train_sentences.copy() + corpus.test_sentences.copy()
    num_train_and_dev = len(corpus.train_sentences)
    vectorizer = TfidfVectorizer("english")
    features_total = vectorizer.fit_transform(textFeatures)
    features = features_total[0:num_train_and_dev]
    test_features = features_total[num_train_and_dev:]
    features_train, features_test, labels_train, labels_test = train_test_split(features, corpus.labels, test_size=0.3,
                                                                                random_state=111)
    if mode == 'svm':
        from sklearn.svm import SVC

        model = SVC(kernel='sigmoid', gamma=1.0)
        model.fit(features_train, labels_train)
        prediction = model.predict(features_test)
        print(accuracy_score(labels_test, prediction))
    elif mode == 'multinomialnb':
        from sklearn.naive_bayes import MultinomialNB

        model = MultinomialNB(alpha=0.2)
        model.fit(features_train, labels_train)
        prediction = model.predict(features_test)
        print(accuracy_score(labels_test, prediction))
    elif mode == 'mlp':
        from sklearn.neural_network.multilayer_perceptron import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(512, 256, 128))
        model.fit(features_train, labels_train)
        prediction = model.predict(features_test)
        print(accuracy_score(labels_test, prediction))
    return model, test_features


def test_classifier(model, test_features, corpus: Corpus):
    predictions = [['SmsId', 'Label']]
    ys = model.predict(test_features)
    ys = list(map(lambda x: 'spam' if x == 1 else 'ham', ys))
    predictions = predictions + np.array([corpus.test_smsids, ys]).transpose().tolist()
    with open('data/submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(predictions)


if __name__ == '__main__':
    corpus = Corpus()
    # model = Model(corpus, num_embeddings=None, embedding_dim=50, hidden_size=128, hidden_dim=64).cuda()
    # optimizer = Adam(model.parameters(), lr=0.001)
    # loss_fn = nn.BCELoss().cuda()
    # dataloaders = {'train': DataLoader(SpamSet(True, corpus, max_len=max_len), batch_size=8, shuffle=True),
    #                'dev': DataLoader(SpamSet(False, corpus, max_len=max_len), batch_size=8, shuffle=False)}
    # train(model, loss_fn, optimizer, dataloaders)
    # test(corpus, model, optimizer)
    model, test_features = train_classifier(corpus, 'mlp')
    test_classifier(model, test_features, corpus)
    #
    # corpus = Corpus_v2()
    # model = Model_v2(corpus, max_len=max_len).cuda()
    # optimizer = Adam(model.parameters(), lr=0.0001)
    # loss_fn = nn.BCELoss().cuda()
    # dataloaders = {'train': DataLoader(SpamSet_v2(True, corpus, max_len=max_len), batch_size=8, shuffle=True),
    #                'dev': DataLoader(SpamSet_v2(False, corpus, max_len=max_len), batch_size=8, shuffle=False)}
    # train_v2(model, loss_fn, optimizer, dataloaders)
    # test_v2(corpus, model, optimizer)
