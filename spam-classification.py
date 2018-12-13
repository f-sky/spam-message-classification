from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from spam_utils_v1 import *
from spam_utils_v2 import *
from spam_utils_v3 import Corpus_v3, Model_v3, SpamSet_v3, train_v3, test_v3
from spam_utils_v4 import train_classifier, test_classifier

num_epochs = 40
max_len = 30


def version1():
    # ---------------version1:embedding rnn------------------
    corpus = Corpus()
    model = Model(corpus, embedding_dim=50, hidden_size=256, hidden_dim=64).cuda()
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss().cuda()
    scheduler = lr_scheduler.StepLR(optimizer, 10)
    dataloaders = {'train': DataLoader(SpamSet(True, corpus, max_len=max_len), batch_size=128, shuffle=True),
                   'dev': DataLoader(SpamSet(False, corpus, max_len=max_len), batch_size=128, shuffle=False)}
    train_v1(model, loss_fn, optimizer, dataloaders, num_epochs, None)
    test_v1(corpus, model, optimizer, max_len=max_len)


def version2():
    # --------------sklearn mlp by pytorch--------------------
    corpus = Corpus_v2()
    model = Model_v2(corpus).cuda()
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss().cuda()
    dataloaders = {'train': DataLoader(SpamSet_v2(True, corpus), batch_size=128, shuffle=True),
                   'dev': DataLoader(SpamSet_v2(False, corpus), batch_size=128, shuffle=False)}
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
    train_v2(model, loss_fn, optimizer, dataloaders, scheduler, num_epochs=num_epochs)
    test_v2(corpus, model, optimizer)


def version3():
    # --------------lstm pretrained weights by pytorch--------------------
    corpus = Corpus_v3('data/glove.6B.300d.txt')
    model = Model_v3(corpus).cuda()
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss().cuda()
    dataloaders = {'train': DataLoader(SpamSet_v3(True, corpus, max_len), batch_size=128, shuffle=True),
                   'dev': DataLoader(SpamSet_v3(False, corpus, max_len), batch_size=128, shuffle=False)}
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
    train_v3(model, loss_fn, optimizer, dataloaders, scheduler=None, num_epochs=num_epochs)
    test_v3(corpus, model, optimizer, max_len=max_len)


def version4():
    # ------------------sklearn-----------
    corpus = Corpus()
    model, test_features = train_classifier(corpus, 'mlp')
    test_classifier(model, test_features, corpus)


if __name__ == '__main__':
    # version1()
    # version2()
    version3()
    # version4()
