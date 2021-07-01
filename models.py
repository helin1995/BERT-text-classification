import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup


# 定义model
class Bert_Model(nn.Module):
    def __init__(self, bert_path, classes=10):
        super(Bert_Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)  # 导入模型超参数
        self.bert = BertModel.from_pretrained(bert_path)  # 加载预训练模型权重
        self.fc = nn.Linear(self.config.hidden_size, classes)  # 直接分类

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]  # pooled output: [bs, config.hidden_size]
        logit = self.fc(out_pool)  # [bs, classes]
        return logit
