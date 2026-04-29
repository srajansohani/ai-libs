import torch
import torch.nn as nn

class SelfAttentionLayer(nn.Module):
    def __init__(self,input_dim,output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Wq = nn.Linear(input_dim, output_dim)
        self.Wk = nn.Linear(input_dim, output_dim)
        self.Wv = nn.Linear(input_dim, output_dim)
    
   
    
    def forward(self,input):
        Q = self.Wq(input)
        K = self.Wk(input)
        V = self.Wv(input)

        attention_score = Q @ K 
        attention_weights = torch.softmax(attention_score/K.shape[-1] ** 0.5, dim=-1)
        contextual_embeddings = attention_weights @ V

        return contextual_embeddings
    


    ##Instead of matrix we can say all of Wq, Wk, and Wv are a linear layered NN

