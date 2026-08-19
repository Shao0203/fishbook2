import numpy as np
import matplotlib.pyplot as plt
from dataset import ptb
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD


def preprocess(text):
    words = text.lower().replace('.', ' .').split()
    id_to_word = {id: word for id, word in enumerate(dict.fromkeys(words))}
    word_to_id = {word: id for id, word in id_to_word.items()}
    corpus = np.array([word_to_id[word] for word in words])

    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print(f'{query} is not found')
        return

    print(f'\n[query] {query}')
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1*similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(f'{id_to_word[i]}: {similarity[i]}')
        count += 1
        if count >= top:
            return


def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print(f'{cnt/total:.2%} done')
    return M


def ppmi_fast(C, eps=1e-8):
    N = np.sum(C)
    S = np.sum(C, axis=0)
    # 全部向量化，一次算完
    M = np.log2(C * N / (np.outer(S, S)) + eps)
    M = np.maximum(M, 0)
    return M.astype(np.float32)


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
C = create_co_matrix(corpus, len(word_to_id))

# print(C)
# [[0 1 0 0 0 0 0]
# [1 0 1 0 1 1 0]
# [0 1 0 1 0 0 0]
# [0 0 1 0 1 0 0]
# [0 1 0 1 0 0 0]
# [0 1 0 0 0 0 1]
# [0 0 0 0 0 1 0]]


# x, y = C[word_to_id['you']], C[word_to_id['i']]
# print(cos_similarity(x, y)) # 0.7071067691154799


# most_similar('you', word_to_id, id_to_word, C)
# [query] you
# goodbye: 0.7071067691154799
# i: 0.7071067691154799
# hello: 0.7071067691154799
# say: 0.0
# and: 0.0


W = ppmi(C)
np.set_printoptions(precision=3)
# print(W)
# [[0.  1.807 0.  0.  0.  0.  0.  ]
# [1.807 0.  0.807 0.  0.807 0.807 0.  ]
# [0.  0.807 0.  1.807 0.  0.  0.  ]
# [0.  0.  1.807 0.  1.807 0.  0.  ]
# [0.  0.807 0.  1.807 0.  0.  0.  ]
# [0.  0.807 0.  0.  0.  0.  2.807]
# [0.  0.  0.  0.  0.  2.807 0.  ]]


# SVD dimentionality reduction, 奇异值分解（Singular Value Decomposition）
U, S, V = np.linalg.svd(W)
# print(C[0]) # [0 1 0 0 0 0 0]
# print(W[0]) # [0.  1.807 0.  0.  0.  0.  0.  ]
# print(U[0]) # [ 0.000e+00 3.409e-01 -3.886e-16 -1.205e-01 9.323e-01 -1.110e-16 -1.467e-16]
# print(U[0, :2])# [0.  0.341]
# for word, word_id in word_to_id.items():
#   plt.annotate(word, (U[word_id, 1], U[word_id, 0]))
# plt.scatter(U[:, 1], U[:, 0], alpha=0.5)
# plt.show()


# use PTB dataset
corpus, word_to_id, id_to_word = ptb.load_data('train')
# print('corpus size:', len(corpus))  # 929589
# print('corpus[:30]:', corpus[:30])  # [ 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29]
# print()
# print('id_to_word[0]:', id_to_word[0])  # aer
# print('id_to_word[1]:', id_to_word[1])  # banknote
# print('id_to_word[2]:', id_to_word[2])  # berlitz
# print()
# print("word_to_id['car']:", word_to_id['car'])  # 3856
# print("word_to_id['happy']:", word_to_id['happy'])  # 4428
# print("word_to_id['lexus']:", word_to_id['lexus'])  # 7426

window_size = 2
wordvec_size = 100
vocab_size = len(word_to_id)
print('counting  co-occurrence ...')
C = create_co_matrix(corpus, vocab_size, window_size)
print('calculating PPMI ...')
# W = ppmi(C, verbose=True)
W = ppmi_fast(C)
print('calculating SVD ...')
U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
# svd = TruncatedSVD(n_components=wordvec_size)
# U = svd.fit_transform(W)
word_vecs = U[:, :wordvec_size]
querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
# [query] you
# i: 0.6683909296989441
# we: 0.6234626173973083
# do: 0.5560856461524963
# anybody: 0.5408862829208374
# 'll: 0.5313359498977661

# [query] year
# month: 0.6911765336990356
# earlier: 0.6645152568817139
# last: 0.6481361389160156
# next: 0.5945568680763245
# quarter: 0.5872685313224792

# [query] car
# auto: 0.6674804091453552
# luxury: 0.6181908249855042
# lexus: 0.5457295775413513
# truck: 0.5372719764709473
# vehicle: 0.5241742730140686

# [query] toyota
# motor: 0.71128910779953
# nissan: 0.6825441718101501
# motors: 0.6393506526947021
# honda: 0.6002441048622131
# lexus: 0.5943219661712646


# 用TruncatedSVD的结果分数会更高（from sklearn.decomposition import TruncatedSVD）
# [query] you
# i: 0.8387488126754761
# we: 0.8179115056991577
# do: 0.7789794206619263
# 'll: 0.7741351127624512
# 'd: 0.7482922673225403

# [query] year
# earlier: 0.8136488795280457
# month: 0.7911677956581116
# last: 0.7895300388336182
# quarter: 0.7681536674499512
# next: 0.7296322584152222

# [query] car
# auto: 0.7628152370452881
# luxury: 0.7389827370643616
# cars: 0.6834427118301392
# truck: 0.6546990871429443
# domestic: 0.6534252166748047

# [query] toyota
# motor: 0.8018085956573486
# nissan: 0.7550964951515198
# motors: 0.7509176731109619
# lexus: 0.6758137941360474
# mazda: 0.6726537942886353
