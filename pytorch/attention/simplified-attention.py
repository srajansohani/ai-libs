

# import tiktoken
# text = "your journey starts with one step"

# ## Tokenization 

# tokens = text.split(" ")


# print(tokens)

# vocab = {}

# def tokenize(text):
#     text = text.lower()
#     text = text.split(" ")    
#     for i in text:
#         if i not in vocab:
#             vocab[i] = len(vocab) 

#     tokens = []
#     for j in text:
#         tokens.append(vocab[j])
#     return tokens


# tokens = tokenize(text)


# def embedding(tokens, embedding_dim):
#     embedded_tokens = []
#     for token in tokens:
#         embedded_token = [0] * embedding_dim
#         embedded_token[token % embedding_dim] = 1
#         embedded_tokens.append(embedded_token)
#     return embedded_tokens

# embeddings = embedding(tokens, embedding_dim=3)

# print(embeddings)

##Showing example code above that we can also generate token and embeddings but below will not be using it as we need to focus only on attention and not on tokenization and embedding. not using any model as well 



import torch
# Embedding (Considering 3 dimensional embedding)




text = "your journey starts with one step"

inputs = torch.tensor([
    [0.43,0.15,0.89], #your   (x^1)
    [0.55,0.87,0.66], #journey (x^2)
    [0.57,0.85,0.64], #starts  (x^3)
    [0.22,0.58,0.33], #with    ((x^4)
    [0.77,0.25,0.10], #one     ()
    [0.05,0.80,0.55]# step
])




attention_scores = inputs @ inputs.T

print(attention_scores,type(attention_scores))


##Converting Attetnion scores to weights 

atttention_weights = torch.softmax(attention_scores,dim=-1)
print(atttention_weights,type(atttention_weights))