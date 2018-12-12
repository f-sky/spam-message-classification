from base_utils import *
from spam_utils import *
import matplotlib.pyplot as plt
from torch.autograd import backward
from torch.optim import lr_scheduler, Adam
from torch.utils.data import Dataset, DataLoader

corpus = Corpus()
model = Model(corpus, num_embeddings=None, embedding_dim=50, hidden_size=128, hidden_dim=64).cuda()
optimizer = Adam(model.parameters())


