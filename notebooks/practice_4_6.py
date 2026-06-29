'''
1) pip install jieba text2vec gensim
2) 安装词向量，https://modelscope.cn/models/lili666/text2vec-word2vec-tencent-chinese/summary
    打开【模型文件】
    找到 light_Tencent_AILab_ChineseEmbedding.bin, 点击下载
    下载后，放在 notebooks/data/p1ch4 文件夹内

Word2vec 算法演示
https://ronxin.github.io/wevi/
'''

# https://pypi.org/project/text2vec/
# https://github.com/shibing624/text2vec/blob/master/text2vec/word2vec.py
from text2vec import Word2Vec
import torch
from torch import nn
cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
torch.set_printoptions(edgeitems=2, threshold=30)


if __name__ == "__main__":
    w2v_model = Word2Vec("./data/p1ch4/light_Tencent_AILab_ChineseEmbedding.bin", {
        "binary": True
    })

    '''
    将一个词转化为张量
    '''
    word = "苹果"
    word_v = w2v_model.encode(word)
    print(type(word_v))
    print(word_v.shape)

    word2 = "橘子"
    word2_v = w2v_model.encode(word2)

    w_t = torch.from_numpy(word_v)
    w2_t = torch.from_numpy(word2_v)

    print("苹果和橘子的距离：")
    print(cos(w_t, w2_t))

    word3 = "篮球"
    word3_v = w2v_model.encode(word3)
    w3_t = torch.from_numpy(word3_v)
    print("苹果和%s的距离：" % word3)
    print(cos(w_t, w3_t))

    '''
    将一句话转化为张量
    '''
    sent = '花呗更改绑定银行卡'
    sentence_embedding = w2v_model.encode(sent)
    print(sentence_embedding.shape)

    sents = ['花呗更改绑定银行卡', '花呗更改']
    sentence_embeddings = w2v_model.encode(sents)
    for (sent, v) in zip(sents, sentence_embeddings):
        v = torch.from_numpy(v)
        print(sent)
        print(v.shape)
        print(v)
