'''
安装包

pip install jieba text2vec gensim

下载词向量
https://modelscope.cn/models/lili666/text2vec-word2vec-tencent-chinese/summary

下载文件放在 --> 
* data/p1ch4/light_Tencent_AILab_ChineseEmbedding.bin

Word2vec 算法演示
https://ronxin.github.io/wevi/
'''

import jieba

# https://pypi.org/project/text2vec/
# https://github.com/shibing624/text2vec/blob/master/text2vec/word2vec.py
from text2vec import Word2Vec


def compute_emb(model):
    # Embed a list of sentences
    sentences = [
        '卡',
        '银行卡',
        '如何更换花呗绑定银行卡',
        '花呗更改绑定银行卡',
        'This framework generates embeddings for each input sentence',
        'Sentences are passed as a list of string.',
        'The quick brown fox jumps over the lazy dog.',
        '敏捷的棕色狐狸跳过了懒狗',
    ]
    sentence_embeddings = model.encode(
        sentences, show_progress_bar=True, normalize_embeddings=True)
    print(type(sentence_embeddings), sentence_embeddings.shape)

    # The result is a list of sentence embeddings as numpy arrays
    for sentence, embedding in zip(sentences, sentence_embeddings):
        print("Sentence:", sentence)
        print("Embedding shape:", embedding.shape)
        print("Embedding head:", embedding[:10])
        print()


if __name__ == "__main__":
    # 中文词向量模型(word2vec)，中文字面匹配任务和冷启动适用
    w2v_model = Word2Vec(
        "./data/p1ch4/light_Tencent_AILab_ChineseEmbedding.bin", {
            "binary": True
        })

    # 打印描述
    print(str(w2v_model))
    print("*" * 80)

    # 将单词转化为向量
    w = "银行"
    w_emb = w2v_model.encode(w)
    print("w_emb type", type(w_emb))
    print("w_emb shape", w_emb.shape)

    # 将句子转化为向量
    compute_emb(w2v_model)
