import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup

max_len = 30  # 文本最大长度
bert_path = './chinese_roberta_wwm_ext_pytorch'
tokenizer = BertTokenizer.from_pretrained(bert_path)  # 分词器


def get_input_data(data_path):
    """
    get input_ids, token_type_ids, input_mask encoding from input text data
    """
    input_ids, input_masks, input_types, = [], [], []  # input char ids, segment type ids,  attention mask
    labels = []  # 标签
    with open(data_path, encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f)):
            title, y = line.strip().split('\t')

            # encode_plus会输出一个字典，分别为'input_ids', 'token_type_ids', 'attention_mask'对应的编码
            # input_ids：输入的词序列对应的id序列（首个字符是[CLS]，最后一个字符是[SEP]，根据参数会短则补齐0->[PAD]，长则截断）
            # token_type_ids：第一个句子值为0，第二个句子为1
            # attention_mask：与序列长度对应的片段值为1，padding部分为0
            # 根据参数会短则补齐，长则切断
            encode_dict = tokenizer.encode_plus(text=title, max_length=max_len,
                                                padding='max_length', truncation=True)

            input_ids.append(encode_dict['input_ids'])
            input_types.append(encode_dict['token_type_ids'])
            input_masks.append(encode_dict['attention_mask'])

            labels.append(int(y))

    input_ids, input_types, input_masks = np.array(input_ids), np.array(input_types), np.array(input_masks)
    labels = np.array(labels)
    return input_ids, input_types, input_masks, labels


def get_parameter_number(model):
    #  打印模型参数量
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'Total parameters: {}, Trainable parameters: {}'.format(total_num, trainable_num)


# 评估模型性能，在验证集上
def evaluate(model, data_loader, device):
    model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for idx, (ids, att, tpe, y) in (enumerate(data_loader)):
            y_pred = model(ids.to(device), att.to(device), tpe.to(device))
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
            val_true.extend(y.squeeze().cpu().numpy().tolist())

    return accuracy_score(val_true, val_pred)  # 返回accuracy


# 测试集没有标签，需要预测提交
def predict(model, data_loader, device):
    model.eval()
    val_pred = []
    with torch.no_grad():
        for idx, (ids, att, tpe) in tqdm(enumerate(data_loader)):
            y_pred = model(ids.to(device), att.to(device), tpe.to(device))
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
    return val_pred


def train_and_eval(model, train_loader, valid_loader,
                   optimizer, scheduler, device, epoch):
    best_acc = 0.0
    patience = 0
    criterion = nn.CrossEntropyLoss()
    for i in range(epoch):
        """训练模型"""
        start = time.time()
        model.train()
        print("***** Running training epoch {} *****".format(i + 1))
        train_loss_sum = 0.0
        for idx, (ids, att, tpe, y) in enumerate(train_loader):
            ids, att, tpe, y = ids.to(device), att.to(device), tpe.to(device), y.to(device)
            y_pred = model(ids, att, tpe)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # 学习率变化

            train_loss_sum += loss.item()
            if (idx + 1) % (len(train_loader) // 5) == 0:  # 只打印五次结果
                print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}".format(
                    i + 1, idx + 1, len(train_loader), train_loss_sum / (idx + 1), time.time() - start))
                # print("Learning rate = {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

        """验证模型"""
        model.eval()
        acc = evaluate(model, valid_loader, device)  # 验证模型的性能
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_bert_model.pth")

        print("current acc is {:.4f}, best acc is {:.4f}".format(acc, best_acc))
        print("time costed = {}s \n".format(round(time.time() - start, 5)))
