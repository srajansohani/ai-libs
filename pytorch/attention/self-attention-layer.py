import torch

class SelfAttentionLayer:
    def __init__(self,input_dim,output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initialize_weights()
    
    def initialize_weights(self):
        self.Wq = torch.randn(self.input_dim, self.output_dim) * (2 / self.input_dim) ** 0.5
        self.Wk = torch.randn(self.input_dim, self.output_dim) * (2 / self.input_dim) ** 0.5
        self.Wv = torch.randn(self.input_dim, self.output_dim) * (2 / self.input_dim) ** 0.5

        self.Wq.requires_grad_()
        self.Wk.requires_grad_()
        self.Wv.requires_grad_()
    
    def forward(self,input):
        Q = input @ self.Wq
        K = input @ self.Wk
        V = input @ self.Wv

        attention_score = Q @ K 
        attention_weights = torch.softmax(attention_score/self.input_dim ** 0.5, dim=-1)
        contextual_embeddings = attention_weights @ V

        return contextual_embeddings