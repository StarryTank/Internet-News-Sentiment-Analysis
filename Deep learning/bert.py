# coding: UTF-8
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
import numpy as np
from tqdm import tqdm
from torch.utils.data import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

#bert_path = "chinese_roberta_wwm_ext_pytorch/"
#tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")  # 初始化分词器

input_ids = []  # input char ids
input_types = []  # segment ids
input_masks = []  # attention mask
label = []  # 标签
pad_size = 32  # 也称为 max_len (前期统计分析，文本长度最大值为38，取32即可覆盖99%)()

with open("CS.txt", encoding='utf-8') as f:
    for i, l in tqdm(enumerate(f)):
        #print(l)
        x1, y = l.strip().rsplit(',',1)

        #print(x1)
        #print(y)
        x1 = tokenizer.tokenize(x1)
        tokens = ["[CLS]"] + x1 + ["[SEP]"]

        # 得到input_id, seg_id, att_mask
        ids = tokenizer.convert_tokens_to_ids(tokens)
        #print(ids)
        types = [0] * (len(ids))
        #print(types)
        masks = [1] * len(ids)
        #print(masks)
        # 短则补齐，长则切断
        if len(ids) < pad_size:
            types = types + [1] * (pad_size - len(ids))  # mask部分 segment置为1
            print(types)
            masks = masks + [0] * (pad_size - len(ids))
            ids = ids + [0] * (pad_size - len(ids))
        else:
            types = types[:pad_size]
            masks = masks[:pad_size]
            ids = ids[:pad_size]
        input_ids.append(ids)
        input_types.append(types)
        input_masks.append(masks)
        #         print(len(ids), len(masks), len(types))
        assert len(ids) == len(masks) == len(types) == pad_size
        label.append([int(float(y))])

# 随机打乱索引
random_order = list(range(len(input_ids)))
np.random.seed(2020)  # 固定种子
np.random.shuffle(random_order)
print(random_order[:10])

# 4:1 划分训练集和测试集
input_ids_train = np.array([input_ids[i] for i in random_order[:int(len(input_ids) * 0.8)]])
input_types_train = np.array([input_types[i] for i in random_order[:int(len(input_ids) * 0.8)]])
input_masks_train = np.array([input_masks[i] for i in random_order[:int(len(input_ids) * 0.8)]])
y_train = np.array([label[i] for i in random_order[:int(len(input_ids) * 0.8)]])
print(input_ids_train.shape, input_types_train.shape, input_masks_train.shape, y_train.shape)

input_ids_test = np.array([input_ids[i] for i in random_order[int(len(input_ids) * 0.8):]])
input_types_test = np.array([input_types[i] for i in random_order[int(len(input_ids) * 0.8):]])
input_masks_test = np.array([input_masks[i] for i in random_order[int(len(input_ids) * 0.8):]])
y_test = np.array([label[i] for i in random_order[int(len(input_ids) * 0.8):]])
print(input_ids_test.shape, input_types_test.shape, input_masks_test.shape, y_test.shape)
#print(y_test)
BATCH_SIZE = 16
train_data = TensorDataset(torch.LongTensor(input_ids_train),
                           torch.LongTensor(input_types_train),
                           torch.LongTensor(input_masks_train),
                           torch.LongTensor(y_train))
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(torch.LongTensor(input_ids_test),
                          torch.LongTensor(input_types_test),
                          torch.LongTensor(input_masks_test),
                          torch.LongTensor(y_test))
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")  # /bert_pretrain/
        for param in self.bert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度
        self.fc = nn.Linear(768, 3)  # 768 -> 2

    def forward(self, x):
        context = x[0]  # 输入的句子   (ids, seq_len, mask)
        types = x[1]
        mask = x[2]  # 对padding部分进行mask，和句子相同size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, token_type_ids=types,
                              attention_mask=mask,
                              output_all_encoded_layers=False)  # 控制是否输出所有encoder层的结果
        out = self.fc(pooled)  # 得到3分类
        return out

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(DEVICE)
print(model)
train_losses = []
test_losses = []
accs = []
"""param_optimizer = list(model.named_parameters())  # 模型参数名字列表
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

NUM_EPOCHS = 3
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-5,
                     warmup=0.05,
                     t_total=len(train_loader) * NUM_EPOCHS
                     )
"""
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)   # 简单起见，可用这一行代码完事
NUM_EPOCHS = 3

def train(model, device, train_loader, optimizer, epoch) -> object:  # 训练模型
    model.train()
    best_acc = 0.0
    for batch_idx, (x1, x2, x3, y) in enumerate(train_loader):
        start_time = time.time()
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        y_pred = model([x1, x2, x3])  # 得到预测结果
        model.zero_grad()  # 梯度清零
        loss = F.cross_entropy(y_pred, y.squeeze())  # 得到loss
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 100 == 0:  # 打印loss
            print('Train Epoch: {} [{}/{} ({:.2f}%)]tLoss: {:.6f}'.format(epoch, (batch_idx + 1) * len(x1),
                                                                          len(train_loader.dataset),
                                                                          100. * batch_idx / len(train_loader),
                                                                          loss.item()))  # 记得为loss.item()
        train_losses.append(loss.item())


def test(model, device, test_loader):  # 测试模型, 得到测试集评估结果
    model.eval()
    val_pred=[]
    test_loss = 0.0
    acc = 0
    for batch_idx, (x1, x2, x3, y) in enumerate(test_loader):
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        with torch.no_grad():
            y_ = model([x1, x2, x3])
        test_loss += F.cross_entropy(y_, y.squeeze())
        pred = y_.max(-1, keepdim=True)[1]  # .max(): 2输出，分别为最大值和最大值的index
        for i in pred:
            val_pred.append(i.cpu().numpy())
        acc += pred.eq(y.view_as(pred)).sum().item()  # 记得加item()
    test_loss /= len(test_loader)
    print('nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, acc, len(test_loader.dataset),
        100. * acc / len(test_loader.dataset)))

    test_losses.append(test_loss)
    accs.append(acc / len(test_loader.dataset))
    #确定F1值
    p = precision_score(y_test, val_pred, average='macro')
    r = recall_score(y_test, val_pred, average='macro')
    f1 = f1_score(y_test, val_pred, average='macro')

    print("精确率：", p)
    print("召回率：", r)
    print("F1值：", f1)
    return acc / len(test_loader.dataset)

best_acc = 0.0
PATH = 'roberta_model.pth'  # 定义模型保存路径
for epoch in range(1, NUM_EPOCHS+1):  # 3个epoch
    train(model, DEVICE, train_loader, optimizer, epoch)
    acc = test(model, DEVICE, test_loader)
    if best_acc < acc:
        best_acc = acc
        torch.save(model, PATH)  # 保存最优模型
    print("acc is: {:.4f}, best acc is {:.4f}n".format(acc, best_acc))

def draw_picture(train_losses,test_losses,accs):
    #画图
    a = [i for i in train_losses]
    b = [i for i in test_losses]
    c = [i for i in accs]
    #训练集
    plt.figure()
    plt.plot(a, '-', label='Train Loss')
    plt.xlabel(xlabel="Epoch")
    plt.ylabel(ylabel='Loss')
    plt.legend()
    plt.show()
    #测试集
    plt.figure()
    plt.plot(b, '-', label='Validation Loss')
    plt.plot(c, '-', label='Accuracy')
    plt.xlabel(xlabel="Epoch")
    plt.xlim(0,2,1)
    plt.ylim(0,1)
    plt.legend()
    plt.show()

#draw_picture(train_losses,test_losses,accs)

#model.load_state_dict(torch.load("roberta_model.pth"))
#acc = test(model, DEVICE, test_loader)



# 如果打比赛的话，下面代码也可参考
"""
# 测试集提交
PATH = "roberta_model.pth"
model.load_state_dict(torch.load(PATH))
def test_for_submit(model, device, test_loader):    # 测试模型
    model.eval()
    preds = []
    for batch_idx, (x1,x2,x3) in tqdm(enumerate(test_loader)):
        x1,x2,x3 = x1.to(device), x2.to(device), x3.to(device)
        with torch.no_grad():
            y_ = model([x1,x2,x3])
        pred = y_.max(-1, keepdim=True)[1].squeeze().cpu().tolist()   
        # .max() 2输出，分别为最大值和最大值的index
        preds.extend(pred) 
    return preds 
preds = test_for_submit(model, DEVICE, test_loader)
"""

