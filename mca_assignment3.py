
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.corpus import abc
import nltk
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

window_context = 2

class CBOWModel(nn.Module):
    def __init__(self, len_vocab,len_embed):
        super(CBOWModel, self).__init__()
        self.embed = nn.Embedding(len_vocab, len_embed)
        self.layer = nn.Linear(len_embed, len_vocab)
        
    def forward(self,x):
        temp = torch.mean(self.embed(x),dim=0)
        temp = temp.view((1, -1))
        temp = self.layer(temp)
        temp = F.log_softmax(temp)
        return temp

class SkipGramModel(nn.Module):
    def __init__(self, len_vocab, len_embed):
        super(SkipGramModel, self).__init__()
        self.embed = nn.Embedding(len_vocab, len_embed)
        self.layer = nn.Linear(len_embed,len_vocab)

    def forward(self, center, context):
        temp_center = self.embed(center)
        temp_center = temp_center.view((1, -1))
        temp_context = self.embed(context)
        temp_context = temp_context.view((1, -1))
        temp_context = torch.t(temp_context)
        #s = torch.mm(temp_center,temp_context)
        log_probs = F.logsigmoid(temp_center)
        return log_probs
    
    def predict(self,word):
      return self.embed(word)

def plot_tsne_skip(skip_data,skip_model,ser):
  vectors = []
  words = []
  for in_w,out_w in skip_data:
    if in_w in words:
      continue
    in_w_var = Variable(torch.LongTensor([w2i[in_w]]))
    out_w_var = Variable(torch.LongTensor([w2i[out_w]]))
    log_probs = skip_model.predict(in_w_var, out_w_var)
    vectors.append(log_probs.view((-1)).detach().numpy())
    words.append(in_w)
  #print(vectors[0])
  vectors_2 = TSNE(n_components=2).fit_transform(vectors)
  plt.figure()
  sns.set(rc={'figure.figsize':(15,15)})
  #print(vectors_2)
  #print(words)
  s = sns.scatterplot(vectors_2[:,0], vectors_2[:,1],palette = 'Blues')
  label_point(vectors_2[:,0],vectors_2[:,1],words,s)
  plt.savefig(str(ser)+'.png')
  plt.close()

def label_point(x, y, val, ax):
    a = pd.concat({'x': pd.Series(x), 'y': pd.Series(y), 'val': pd.Series(val)}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']))

def skipgram_dataset(text):
    dataset = []
    temp = 2
    start = temp
    end = len(text)-temp
    for i in range(start, end):
        dataset.append((text[i], text[i-temp]))
        dataset.append((text[i], text[i-(temp-1)]))
        dataset.append((text[i], text[i+(temp-1)]))
        dataset.append((text[i], text[i+temp]))
    return dataset

def cbow_dataset(text):
    dataset = []
    temp = 2
    start = temp
    end = len(text)-temp
    for i in range(start, end):
        context = [text[i - temp]]
        context.append(text[i - (temp-1)])
        context.append(text[i + (temp-1)])
        context.append(text[i + temp])
        target = text[i]
        dataset.append((context, target))
    return dataset

def train_cbow(model):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total = .0
        for x,y in cbow_train:
            model.zero_grad()
            index = [w2i[w] for w in x]
            tensor = Variable(torch.LongTensor(index))
            log_probs = model(tensor)
            compare = Variable(torch.LongTensor([w2i[y]]))
            loss = loss_fn(log_probs, compare)
            loss.backward()
            optimizer.step()
            total += loss.data.item()
        print(epoch,total_loss)
    return model

def train_skipgram(model):
    optimizer = optim.SGD(model.parameters(), lr=lr)    
    for epoch in range(epochs):
        total= .0
        for x, y in skipgram_train:
            temp_x = Variable(torch.LongTensor([w2i[x]]))
            temp_y = Variable(torch.LongTensor([w2i[y]]))
            model.zero_grad()
            log_probs = model(temp_x, temp_y)
            compare = Variable(torch.Tensor([1]))
            loss = loss_fn(log_probs[0],compare)          
            loss.backward()
            optimizer.step()
            total += loss.data.item()
        print(epoch,total_loss)
        plot_tsne_skip(skipgram_train[:1000],model,epoch)
    return model

nltk.download('abc')
text = list(abc.words())
vocab = set(text)
vocab_size = len(vocab)
embd_size = 50
lr = 0.1
epochs = 50
hidden_size = 100
pt = 0
for word in vocab:
    w2i[word] = pt
    i2w[pt] = word
    pt+=1
subset = text[:5000]
cbow_train = cbow_dataset(subset)
skipgram_train = skipgram_dataset(subset)
loss_fn = nn.NLLLoss()
cbow = CBOWModel(vocab_size, embd_size)
cbow_model = train_cbow(cbow)
torch.save(cbow.state_dict(), 'cbow.pth')
loss_fn = nn.MSELoss()
skip_gram = SkipGramModel(vocab_size, embd_size)
sg_model= train_skipgram(skip_gram)
torch.save(skip_gram.state_dict(), 'skipgram.pth')
