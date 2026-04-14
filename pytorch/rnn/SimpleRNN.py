from torch import nn

class SimpleRNN(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim=50)
        self.rnn = nn.RNN(input_size=50, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64,vocab_size)

    def forward(self,x):
        x = self.embedding(x)
        hidden,final = self.rnn(x)
        output = self.fc(final)
        return output
        
    