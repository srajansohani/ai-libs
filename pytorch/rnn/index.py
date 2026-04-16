#Using RNN


#Load Text Data


#eng_text -> tokens -> sentences -> embeddings -> rnn -> output, #inout_layer = embedding_dimension

import pandas as pd;
import torch;
from torch.utils.data import Dataset,DataLoader
from SimpleRNN import SimpleRNN



data = pd.read_csv("./data/100_Unique_QA_Dataset.csv")


##Tokenization

def tokenize(text):
    text = text.lower()
    text = text.replace('?', '')
    text = text.replace('"', "")
    return text.split()

vocab = {
    '<UNK>':0
}

def build_vocab(row):
  tokenized_question = tokenize(row['question'])
  tokenized_answer = tokenize(row['answer'])
  merged_tokens = tokenized_question + tokenized_answer
  for token in merged_tokens:
    if token not in vocab:

        vocab[token] = len(vocab)

def text_to_indices(text,vocab):
   indexed_text = []

   for token in tokenize(text):
       if token in vocab:
           indexed_text.append(vocab[token])
       else:
           indexed_text.append(vocab['<UNK>'])
   return indexed_text

class QuestionDataset(Dataset):
    def __init__(self,df,vocab):
        self.df = df
        self.vocab = vocab
    
    def __len__(self):
        return self.df.shape[0]
    

    def __getitem__(self, index):
       numerica_question =  torch.tensor(text_to_indices(self.df.iloc[index]['question'],vocab=vocab))
       numerical_answer = torch.tensor(text_to_indices(self.df.iloc[index]['answer'],vocab=vocab))
       return numerica_question, numerical_answer

data.apply(build_vocab, axis=1)
dataset = QuestionDataset(data,vocab)

dataloader = DataLoader(dataset=dataset,batch_size=1,shuffle=True)

model = SimpleRNN(len(vocab))
learning_rate = 0.001
 
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 50

for epoch in range(epochs):
    total_loss = 0
    for question,answer in dataloader:
        optimizer.zero_grad()
        prediction = model(question)
        prediction = prediction.view(-1, prediction.shape[-1])
        answer = answer.view(-1) 
        loss = criterion(prediction,answer)

       
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss:", total_loss)


