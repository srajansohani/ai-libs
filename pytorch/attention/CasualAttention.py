

## to implement causual attention// Masked Attention, 
# we need to make sure that the model can only attend to previous tokens in the sequence, not future tokens. 
# This is typically done by applying a mask to the attention scores.

import torch
import torch.nn as nn

class CausualAttentionLayer(nn.Module):
    def __init__(self,input_dim,output_dim):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Wq = nn.Linear(input_dim, output_dim)
        self.Wk = nn.Linear(input_dim, output_dim)
        self.Wv = nn.Linear(input_dim, output_dim)
    
   
    
    def forward(self,input):
        Q = self.Wq(input)
        K = self.Wk(input)
        V = self.Wv(input)

        attention_score = Q @ K.T 

      
        context_length = attention_score.shape[0]
        ##Main step 
        mask_simple = torch.tril(torch.ones()) ## this will create a lower triangular matrix of ones with the same shape as attention_weights
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1) ## this will create an upper triangular matrix of ones with zeros on the diagonal, which will be used to mask out the future tokens
        masked = attention_score.masked_fill(mask.bool(), float('-inf')) ## this will replace the values in attention_score with -inf where the mask is 1, effectively zeroing out the weights for future tokens when we apply softmax
        
        attention_weights = torch.softmax(masked/K.shape[-1] ** 0.5, dim=-1) ## this will apply softmax to the masked attention scores, giving us the attention weights that sum to 1 for each token in the sequence
        
        ##Using dropout in attention matrix to prevent overfitting
        dropout = nn.Dropout(0.5)

        attention_matrix = dropout(attention_weights)  # Apply dropout to the attention weights
        contextual_embeddings = attention_matrix @ V

        return contextual_embeddings
    


    ##Instead of matrix we can say all of Wq, Wk, and Wv are a linear layered NN



