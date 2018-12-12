import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torch.autograd import backward
from torch.optim import lr_scheduler, Adam
from torch.utils.data import Dataset, DataLoader

from base_utils import load_model
from spam_utils import *

num_epochs = 2
max_len = 30


def train(model, loss_fn, optimizer, dataloaders):
    def compute_acc(x, y):
        # print((x.squeeze() > 0.5).sum()/y.shape[0])
        return accuracy_score(y.squeeze(), x.squeeze() > 0.5)

    def compute_f1(x, y):
        return f1_score(y.squeeze(), x.squeeze() > 0.5)

    fit(model, loss_fn, optimizer, dataloaders, {'accuracy': compute_acc, 'f1': compute_f1}, num_epochs=num_epochs)


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
    with open('submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(predictions)


if __name__ == '__main__':
    corpus = Corpus()
    model = Model(corpus, num_embeddings=None, embedding_dim=50, hidden_size=128, hidden_dim=64).cuda()
    optimizer = Adam(model.parameters())
    loss_fn = nn.BCELoss().cuda()
    dataloaders = {'train': DataLoader(SpamSet(True, corpus, max_len=max_len), batch_size=8, shuffle=True),
                   'dev': DataLoader(SpamSet(False, corpus, max_len=max_len), batch_size=8, shuffle=False)}
    train(model, loss_fn, optimizer, dataloaders)
    # test(corpus, model, optimizer)
