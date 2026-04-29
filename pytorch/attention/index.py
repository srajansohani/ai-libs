import torch
# from CasualAttention import CausualAttentionLayer

# text = "your journey starts with one step"

# inputs = torch.tensor([
#     [0.43,0.15,0.89], #your   (x^1)
#     [0.55,0.87,0.66], #journey (x^2)
#     [0.57,0.85,0.64], #starts  (x^3)
#     [0.22,0.58,0.33], #with    ((x^4)
#     [0.77,0.25,0.10], #one     ()
#     [0.05,0.80,0.55]# step
# ])

# casual_attention_layer = CausualAttentionLayer(3,3)

# output = casual_attention_layer(inputs)

# print(output)

x  =  torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]])
print(x.shape)

y  =  torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]],])

z = x @ y

print(z.shape,z)