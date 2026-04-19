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



Wq = torch.rand((3,3),dtype=float)

Wk = torch.rand((3,3),dtype=float)

Wv = torch.rand((3,3),dtype=float)


Q = inputs @ Wq
K = inputs @ Wk
V = inputs @ Wv

print(Q.shape)
print(K.shape)
print(V.shape) 

attention_score = Q @ K.T ## here we are taking transpose of K matrix to make it compatible for matrix multiplication with Q matrix

contextual_embeddings = ((attention_score) @ V)/(3**0.5) ## here 3 is the dimension of K matrix

