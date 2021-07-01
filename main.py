import pandas as pd
import numpy as np
import json
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup
from utils import *
from models import Bert_Model
import warnings

warnings.filterwarnings('ignore')

bert_path = './chinese_roberta_wwm_ext_pytorch'

input_ids, input_types, input_masks, labels = get_input_data('./THUCNews/data/train.txt')
# 随机打乱索引
idxes = np.arange(input_ids.shape[0])
np.random.seed(0)  # 固定随机种子
np.random.shuffle(idxes)
print(idxes.shape, idxes[:10])

input_ids_train = input_ids[idxes]
input_types_train = input_types[idxes]
input_masks_train = input_masks[idxes]
y_train = labels[idxes]
print(input_ids_train.shape, input_types_train.shape, input_masks_train.shape, y_train.shape)

BATCH_SIZE = 64  # 如果会出现OOM问题，减小它
# 训练集
train_data = TensorDataset(torch.LongTensor(input_ids_train),
                           torch.LongTensor(input_masks_train),
                           torch.LongTensor(input_types_train),
                           torch.LongTensor(y_train))
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

input_ids_dev, input_types_dev, input_masks_dev, labels_dev = get_input_data('./THUCNews/data/dev.txt')
print(input_ids_dev.shape, input_types_dev.shape, input_masks_dev.shape, labels_dev.shape)

# 开发集
dev_data = TensorDataset(torch.LongTensor(input_ids_dev),
                         torch.LongTensor(input_masks_dev),
                         torch.LongTensor(input_types_dev),
                         torch.LongTensor(labels_dev))
dev_sampler = SequentialSampler(dev_data)
dev_loader = DataLoader(dev_data, sampler=dev_sampler, batch_size=BATCH_SIZE)

# 测试集
input_ids_test, input_types_test, input_masks_test, labels_test = get_input_data('./THUCNews/data/test.txt')

test_data = TensorDataset(torch.LongTensor(input_ids_test),
                          torch.LongTensor(input_masks_test),
                          torch.LongTensor(input_types_test))
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 2
model = Bert_Model(bert_path).to(DEVICE)
print(get_parameter_number(model))

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)  # AdamW优化器
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader),
                                            num_training_steps=EPOCHS * len(train_loader))
# 学习率先线性warmup一个epoch，然后cosine式下降。
# 这里给个小提示，一定要加warmup（学习率从0慢慢升上去），要不然你把它去掉试试，基本上收敛不了。

# 训练和验证评估
train_and_eval(model, train_loader, dev_loader, optimizer, scheduler, DEVICE, EPOCHS)

# 加载最优权重对测试集测试
model.load_state_dict(torch.load("best_bert_model.pth"))
pred_test = predict(model, test_loader, DEVICE)
print("\n Test Accuracy = {} \n".format(accuracy_score(labels_test, pred_test)))
print(classification_report(labels_test, pred_test, digits=4))
