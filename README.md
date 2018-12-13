#### version 1 uses embedding-lstm-fcs
#### version 2 uses sparse features and fcs in pytorch
#### classifier version uses sparse features and implemented using sklearn
#### version 3 used embedding-lstm-fcs and fix pretrained word vectors

At present the highest score on test set is:
    300d fixed word vector, following 
    lstm = nn.LSTM(input_size=corpus.embedding_dim, hidden_size=512, batch_first=True)
    fcs = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                 nn.ReLU(),
                                 nn.Dropout(),
                                 nn.Linear(in_features=256, out_features=128),
                                 nn.ReLU(),
                                 nn.Dropout(),
                                 nn.Linear(in_features=128, out_features=1),
                                 nn.Sigmoid()
                                 )
     with Adam 0.001 lr and no scheduler
     precision: 99.552%
 