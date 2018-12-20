from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from spam_utils import Corpus_v3, Model_v3, SpamSet_v3, train_v3, test_v3

num_epochs = 60
max_len = 30

def main():
    # --------------lstm pretrained weights by pytorch--------------------
    corpus = Corpus_v3('data/glove.6B.50d.txt')
    model = Model_v3(corpus).cuda()
    optimizer = Adam(model.parameters(), lr=0.0003)
    loss_fn = nn.BCELoss().cuda()
    dataloaders = {'train': DataLoader(SpamSet_v3(True, corpus, max_len), batch_size=128, shuffle=True),
                   'dev': DataLoader(SpamSet_v3(False, corpus, max_len), batch_size=128, shuffle=False)}
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    train_v3(model, loss_fn, optimizer, dataloaders, scheduler=scheduler, num_epochs=num_epochs)
    test_v3(corpus, model, optimizer, max_len=max_len)


if __name__ == '__main__':
    main()
