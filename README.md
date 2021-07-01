# BERT-text-classification
基于BERT的文本分类
这个代码是采用BERT实现的文本分类代码，也就是采用的BERT中[CLS]对应的输出表示，这个向量是对输入句子的编码。
数据集采用的是THUCNews
chinese_roberta_wwm_ext_pytorch文件夹是bert模型文件，包括config.json，pytorch_model.bin，vocab.txt三个文件。
实验只跑了两轮，最后在测试集上的效果是：acc： 94.3%
