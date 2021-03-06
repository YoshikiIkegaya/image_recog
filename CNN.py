from chainer import Link, Chain, ChainList, Variable, FunctionSet, optimizers, serializers
import chainer.functions as F
import chainer.links as L

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
