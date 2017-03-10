# coding: utf-8
from chainer import Link, Chain, ChainList, Variable, FunctionSet, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import numpy as np
import cv2
import csv
import pickle

# *** 定数 ***
WIDTH, HEIGHT  = 15, 10  # 画像のサイズ
# CNNモデルで使用する定数
input_channel  = 3
output_channel = 12
filter_height  = 3
filter_width   = 3
mid_units      = 180
n_units        = 50
n_label        = 2

# *** モデルの定義 ***
class CNN(Chain):
    """
    Convolutional Neural Network のモデル
        input_channel  : 入力するチャンネル数（通常のカラー画像なら３）
        output_channel : 畳み込み後のチャンネル数
        filter_height  : 畳み込みに使用するフィルターの縦方向のサイズ
        filter_width   : 畳み込みに使用するフィルターの横方向のサイズ
        mid_units      : 全結合の隠れ層１のノード数
        n_units        : 全結合の隠れ層２のノード数
        n_label        : ラベルの出力数（今回は２）
    """
    # *** モデルの構造の定義 ***
    def __init__(self, input_channel, output_channel, filter_height, filter_width, mid_units, n_units, n_label):
        super(CNN, self).__init__(
            conv1 = L.Convolution2D(input_channel, output_channel, (filter_height, filter_width)),
            l1    = L.Linear(mid_units, n_units),
            l2    = L.Linear(n_units, n_label),
        )

    # *** 順方向の計算 ***
    def forward(self, x, t, train=True):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 3)
        h2 = F.dropout(F.relu(self.l1(h1)), train=True)
        y  = self.l2(h2)
        if train:
            return F.softmax_cross_entropy(y, t)
        else:
            return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    # *** 予測 ***
    def predict(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 3)
        h2 = F.dropout(F.relu(self.l1(h1)))
        y  = self.l2(h2)
        return F.softmax(y).data


# *** 訓練データの準備 ***
x_train, y_train = [], []
x_test,  y_test  = [], []
csv_reader = csv.reader(open('data.csv','r'), delimiter=",")
for row in csv_reader:
    # row[0]: ファイル名, row[1]: ラベル
    img = cv2.imread(row[0])  # 画像の読み込み <- OpenCVではNumpyの形式で格納される
    resize_img = cv2.resize(img, (HEIGHT, WIDTH))  # 画像を縮小
    input_img  = np.transpose(resize_img, (2,0,1)) / 255.0 # データ項目入れ替えて、0-1に正規化（チャンネルを一番前に）
    x_train.append(input_img)
    y_train.append(row[1])
train_data  = np.array(x_train).astype(np.float32).reshape(len(x_train), 3, HEIGHT, WIDTH)
train_label = np.array(y_train).astype(np.int32)

# *** モデルの宣言 ***
model = CNN(input_channel, output_channel, filter_height, filter_width, mid_units, n_units, n_label)

# *** 最適化オブジェクト ***
optimizer = optimizers.Adam()
optimizer.setup(model)

# *** 学習 ***
n_epoch = 50
x = Variable(np.asarray(train_data))
t = Variable(np.asarray(train_label))
for i in range(n_epoch):
    optimizer.update(model.forward, x, t)
    loss, accuracy = model.forward(x, t, train=False)
    print("--- ", i+1, "回目の学習 ---")
    print("  損失関数: %1.3f" % loss.data)
    print("  正解率: ", round(accuracy.data * 100), "%")
    print("")

# *** 結果を確認 ***
print(model.predict(x))

# *** 学習したモデルを保存 ***
with open('cnn_classify.pkl', 'wb') as f:
    pickle.dump(model, f)
