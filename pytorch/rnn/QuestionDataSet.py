from torch.utils.data import Dataset

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
        return self.df.iloc[index]['question'], self.df.iloc[index]['answer']