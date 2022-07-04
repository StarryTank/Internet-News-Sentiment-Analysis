import torch
import torch.utils.data as DataSet
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import pandas as pd
import numpy as np
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import sklearn.metrics as sm
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import time
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter


def get_stopwords(stop_word_file):
    with open(stop_word_file,encoding="utf-8") as f:
        stopwords=f.read()
    stopwords_list=stopwords.split('\n')
    custom_stopwords_list=[i for i in stopwords_list]
    return custom_stopwords_list
#获得由停用词组成的列表
stop_words_file = r'hit_stopwords.txt'
stopwords = get_stopwords(stop_words_file)

#将数据进行读取
data=pd.read_csv(r'C:\Users\75282\Documents\Tencent Files\752823729\FileRecv\reviews_score_update.csv',index_col=0,encoding="utf-8")
# 现在是划分数据集
x=data.评论.values.astype('U')
y=data.评分.values

"""# 开始使用TF-IDF进行特征的提取，对分词后的中文语句做向量化。
# 引进TF-IDF的包
TF_Vec = TfidfVectorizer(max_df=0.8,
                         min_df=3,
                         stop_words=frozenset(stopwords)
                         )
t_X = TF_Vec.fit_transform(x)

n_X = t_X.toarray()

"""
vocab = Counter(x)
vocab = sorted(vocab,key = vocab.get,reverse=True) # 将Counter转换为list(排序)
vocab_size = len(vocab)
word2idx =  {b:a for a,b in enumerate(vocab)}
encoded_sentences = [word2idx[word] for word in x]
emb_dim = 500
emb_layer = nn.Embedding(vocab_size,emb_dim)
word_vectors = emb_layer(torch.LongTensor(encoded_sentences))
y1 = np.array(y)


batch_size = 32
x_train, x_test, y_train, y_test = model_selection.train_test_split(word_vectors, y1,test_size=0.1, random_state=1,shuffle=True)
print(x_train.shape)
train_ds = DataSet.TensorDataset(torch.FloatTensor(np.array(x_train.detach(),dtype = float)),torch.LongTensor(np.array(y_train)))
train_loader = DataSet.DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=0)

valid_ds = DataSet.TensorDataset(torch.FloatTensor(np.array(x_test.detach(),dtype = float)),torch.LongTensor(np.array(y_test)))
valid_loader = DataSet.DataLoader(valid_ds,batch_size=batch_size,shuffle=True,num_workers=0)


class LSTMNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,out_size,n_layers=1):
        super(LSTMNetwork,self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.lstm = nn.LSTM(input_size,hidden_size,n_layers,batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size,out_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,input,hidden=None):
        input=input.view(len(x), 1, -1)
        hhh1 = hidden[0]
        output,hhh1 = self.lstm(input,hhh1)
        output = self.dropout(output)
        output = output[:,-1,...]
        output = self.fc(output)

        output = self.softmax(output)
        return output

    def initHidden(self,batch_size):
        out = []
        hidden1 = torch.zeros(1,batch_size,self.hidden_size)
        cell1 = torch.zeros(1,batch_size,self.hidden_size)
        out.append((hidden1,cell1))
        return out

def criterion(outputs,target):
    x= outputs
    loss_f = nn.NLLLoss()
    loss = loss_f(x,target)
    return loss

def rightness(predictions,labels):
    pred = torch.max(predictions.data,1)[1]
    rights = pred.eq(labels.data).sum()
    return rights,len(labels)


lstm = LSTMNetwork(500,20,3)
optimizer = optim.Adam(lstm.parameters(),lr=0.0001)
num_epochs = 45
train_losses = []
valid_losses = []
records = []
y_pred= []
for epoch in range(num_epochs):
    train_loss = []
    for batch,data in enumerate(train_loader):
        lstm.train()
        init_hidden = lstm.initHidden(len(data[0]))
        optimizer.zero_grad()
        x,y = data[0],data[1]
        outputs = lstm(x,init_hidden)
        loss = criterion(outputs,y)
        train_loss.append(loss.data.numpy())
        loss.backward()
        optimizer.step()
        valid_loss = []
        lstm.eval()
        rights = []
        for batch,data in enumerate(valid_loader):
            init_hidden = lstm.initHidden(len(data[0]))
            x,y = Variable(data[0]),Variable(data[1])
            outputs = lstm(x,init_hidden)

            loss = criterion(outputs,y)
            valid_loss.append(loss.data.numpy())

            right = rightness(outputs, y)

            rights.append(right[0]/right[1])
        print('第{}轮，训练Loss：{:.2f},校验Loss：{:.2f},准确度：{:.3f}'.format(epoch,
                                                       np.mean(train_loss),
                                                       np.mean(valid_loss),
                                                       np.mean(rights)
                                                       ))

        records.append([np.mean(train_loss),np.mean(valid_loss),np.mean(rights)])



a = [i[0] for i in records]
b = [i[1] for i in records]
c = [i[2] for i in records]
plt.plot(a,'-',label = 'Train Loss')
plt.plot(b,'-',label = 'Validation Loss')
plt.plot(c,'-',label = 'Accuracy')
plt.xlabel(xlabel="epoch")
plt.legend()
plt.show()




