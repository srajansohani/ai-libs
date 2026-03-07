
# #Tensors are a datastructure to maintain multidimesional array similar as numpy arrays but with the ability to perform operations on GPU and also maintain gradients for backpropagation in neural networks.

# """
#     Dimesion of a tensor is the number of axes it has. For example, a 2D tensor has 2 axes (rows and columns), while a 3D tensor has 3 axes (depth, rows, and columns).
#     A zero dimesional tensor is a scaler which means a normal number.( Loss function computes [y_pred, y_true] and returns a scaler value which is the loss)
#     A one dimesional tensor is a vector which means a list of numbers.
#     A two dimesional tensor is a matrix which means a table of numbers with rows and columns.

# """

# """
#     Uses of Tensors: 
#       1.) Ther ayre efficient for addition, multiplication and other mathematical operations as they can be performed in parallel on GPU. similarly line numpy can do vectorization
# """


# import torch 

# print(torch.__version__)

# if torch.cuda.is_available():
#     print("GPU is available")

# else:
#     print("GPU is not available")



# # ## Creating a Tensors
# #   # uninitialized tensor of size 5
# # a = torch.empty((2,3))

# # type(a)

# # b  = torch.ones((3,5))

# # ##To initialize a tensor with random values
# # c = torch.rand(2,3)

# # ##To maintain same random values
# # torch.manual_seed(100)


# # print("use arrannge ->",torch.arange(0,10,2)) # start, end, step

# # print("linspace ->",torch.linspace(0,1,5)) # start, end, number of points

# # print("eye ->",torch.eye(3)) # identity matrix of size 3x3

# # print("diag ->",torch.diag(torch.tensor([1,2,3]))) # diagonal matrix from a vector

# # #using full
# # print("full ->",torch.full((2,3),7)) # tensor of size 2x3 filled with 7




# # ###Tensor Shapes

# # x = torch.tensor([[1,2,3],[4,5,6]])

# # print("shape of x ->",x.shape) # (2,3)

# # #to copy shape of a tensor
# # y = torch.empty_like(x)

# # #Similarly we can use zeros_like and ones_like to create tensors of same shape filled with zeros and ones respectively
# # y1 = torch.zeros_like(x)
# # y2 = torch.ones_like(x)

# # y3 = torch.tensor([[1.1,2.2,3.7]],dtype=torch.int64) # to specify data type of tensor
# # print(y3)




# #Mathmatical operations on tensors

# #Scaler operations
# x = torch.tensor([[1,2,3],[4,5,6]])
# print("add 2 ->",x + 2) # add 2 to each element of x


# ##Element wise operations
# y = torch.tensor([[7,8,9],[10,11,2]])
# print("x + y ->",x + y) # add x and y element wise
# print("x - y ->",x - y) # subtract y from x element wise
# print("x * y ->",x * y) # multiply x and y element wise
# print("x/y")


# ## Reduction functions

# #Sum b  
class A:
    def __init__(self):
        self.value = 3

a = A()
b = a
b.value = 2
print(a.value)


##  