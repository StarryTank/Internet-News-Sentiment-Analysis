import pandas as pd
import csv
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


#将数据进行读取
data=pd.read_csv(r'C:\\Users\75282\Documents\Tencent Files\752823729\FileRecv\reviews_score_update.csv',index_col=0,encoding="utf-8")
# 现在是划分数据集
x=data.评论.values.astype('U')
y=data.评分.values

#print(np.shape(y))
# random_state 取值，这是为了在不同环境中，保证随机数取值一致，以便验证模型的实际效果。
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y,test_size=0.2, random_state=1)




print(x_train.shape, x_test.shape)
#print(x_test)
# train_x 训练集数据 test_x 测试集数据  train_y训练集的标签 test_y 测试集的标签
#定义函数，从哈工大中文停用词表里面，把停用词作为列表格式保存并返回 在这里加上停用词表是因为TfidfVectorizer和CountVectorizer的函数中
#可以根据提供用词里列表进行去停用词
def get_stopwords(stop_word_file):
    with open(stop_word_file,encoding="utf-8") as f:
        stopwords=f.read()
    stopwords_list=stopwords.split('\n')
    custom_stopwords_list=[i for i in stopwords_list]
    return custom_stopwords_list
#获得由停用词组成的列表
stop_words_file = r'hit_stopwords.txt'
stopwords = get_stopwords(stop_words_file)

'''
使用TfidfVectorizer()和 CountVectorizer()分别对数据进行特征的提取，投放到不同的模型中进行实验
'''
# 开始使用TF-IDF进行特征的提取，对分词后的中文语句做向量化。
# 引进TF-IDF的包
TF_Vec = TfidfVectorizer(max_df=0.8,
                         min_df=3,
                         stop_words=frozenset(stopwords)
                         )
# 拟合数据，将数据准转为标准形式，一般使用在训练集中
train_x_tfvec = TF_Vec.fit_transform(x_train)
# 通过中心化和缩放实现标准化，一般使用在测试集中
test_x_tfvec = TF_Vec.transform(x_test)


'''
使用TF_IDF提取的向量当作数据特征传入模型
'''
#构建模型之前首先将包进行导入
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import time
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier

#XGBoost
model = XGBClassifier(learning_rate =0.01,
n_estimators=140,
max_depth=10,
min_child_weight=10,
gamma=1,
subsample=0.5,
colsample_bytree=0.8,
objective= 'multi:softmax”',
nthread=4,
scale_pos_weight=1,
reg_alpha=0.1,
reg_lambda=0.1)
eval_set = [(test_x_tfvec,y_test)]
model.fit(train_x_tfvec,y_train,early_stopping_rounds=10,eval_metric="merror",eval_set=eval_set,verbose=True)
y_pred = model.predict(test_x_tfvec)
predictions = [round(value) for value in y_pred]
acc = sklearn.metrics.accuracy_score(y_test,predictions)
print("准确率为："+str(acc))

p = precision_score(y_test, y_pred,average='macro')
r = recall_score(y_test, y_pred,average='macro')
f1 = f1_score(y_test, y_pred,average='macro')

print("精确率：",p)
print("召回率：",r)
print("F1值：",f1)
"""
y_test=label_binarize(y_test,classes=[0,1,2])
y_pred=label_binarize(y_pred,classes=[0,1,2])

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0,3):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["macro"], tpr["macro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='blue',
         lw=lw, label='label 0 (area = %0.2f)' % roc_auc[0])
plt.plot(fpr[1], tpr[1], color='red',
         lw=lw, label='label 1 (area = %0.2f)' % roc_auc[1])
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='label 2 (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


"""

"""start_time=time.time()
#创建模型
lr = linear_model.LogisticRegression(penalty='l2', C=1, solver='liblinear', max_iter=1000, multi_class='ovr')
#进行模型的优化，因为一些参数是不确定的，所以就让模型自己在训练中去确定自己的参数 模型的名字也由LR转变为model
model = GridSearchCV(lr, cv=3, param_grid={
        'C': np.logspace(0, 4, 30),
        'penalty': ['l1', 'l2']
    })
#模型拟合tf-idf拿到的数据
model.fit(train_x_tfvec,y_train)
#查看模型自己拟合的最优参数
print('最优参数：', model.best_params_)
#在训练时查看训练集的准确率
pre_train_y=model.predict(train_x_tfvec)
#在训练集上的正确率
train_accracy=accuracy_score(y_train,pre_train_y)
#训练结束查看预测 输入验证集查看预测
pre_test_y=model.predict(test_x_tfvec)
#查看在测试集上的准确率
test_accracy = accuracy_score(y_test,pre_test_y)
print('使用TF-IDF提取特征使用逻辑回归,让模型自适应参数，进行模型优化\n训练集:{0}\n测试集:{1}'.format(train_accracy,test_accracy))
end_time=time.time()
print("使用模型优化的程序运行时间为",end_time-start_time)

p = precision_score(y_test, pre_test_y,average="macro")
r = recall_score(y_test, pre_test_y,average="macro")
f1 = f1_score(y_test, pre_test_y,average="macro")


#tf逻辑斯蒂回归混淆矩阵
cm = sm.confusion_matrix(y_test,pre_test_y)
print('混淆矩阵：',cm)
cp = sm.classification_report(y_test,pre_test_y)
print("分类报告：",cp)

y_test=label_binarize(y_test,classes=[0,1,2])
y_pred=label_binarize(pre_test_y,classes=[0,1,2])

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0,3):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["macro"], tpr["macro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='blue',
         lw=lw, label='label 0 (area = %0.2f)' % roc_auc[0])
plt.plot(fpr[1], tpr[1], color='red',
         lw=lw, label='label 1 (area = %0.2f)' % roc_auc[1])
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='label 2 (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
"""
