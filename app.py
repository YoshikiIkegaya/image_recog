# coding: utf-8
from flask import Flask, Response, render_template, request, make_response, jsonify
from glob import glob
from flaski.database import db_session
from flaski.models import Image
from wtforms import Form, TextField
import csv, io, sys
import numpy as np
from chainer import Link, Chain, ChainList, Variable, FunctionSet, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import cv2
import pickle
from CNN import CNN

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

app = Flask(__name__)

class LabelForm(Form):
    label = TextField('Label')

# *** カテゴリーを予測 ***
def img2category(img_path):
    with open('cnn_classify.pkl', 'rb') as i:
        model = pickle.load(i)
    x_train = []
    img = cv2.imread(img_path)  # 画像の読み込み <- OpenCVではNumpyの形式で格納される
    resize_img = cv2.resize(img, (HEIGHT, WIDTH))  # 画像を縮小
    input_img  = np.transpose(resize_img, (2,0,1)) / 255.0 # データ項目入れ替えて、0-1に正規化（チャンネルを一番前に）
    x_train.append(input_img)
    train_data  = np.array(x_train).astype(np.float32).reshape(len(x_train), 3, HEIGHT, WIDTH)
    x = Variable(np.asarray(train_data))
    return model.predict(x)[0]


# *** トップページ ***
@app.route("/", methods=["GET", "POST"])
def index():
    image = Image.query.filter_by(is_complete=False).first()  # DBからファイルの読み込み
    form  = LabelForm(request.form)
    if request.method == "POST" and form.validate():
        image.label = form.label.data
        image.is_complete = True
        db_session.commit()
        image = Image.query.filter_by(is_complete=False).first()  # DBからファイルの再読み込み
    return render_template("index.html", filename=image.filename, form=form)  # filenameにDBから読み込んだファイル名を渡す


# *** CSVをダウンロード ***
@app.route("/download")
def download():
    # レスポンス用に宣言
    response = make_response()
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = u'attachment; filename=data.csv'
    # ラベル付けが完了しているデータを抽出
    queries = Image.query.filter_by(is_complete=True)
    result = []
    for query in queries:
        result.append((query.filename, query.label))
    # CSV出力用のメソッド
    csv_file = io.StringIO()
    writer = csv.writer(csv_file)
    writer.writerows(result)
    # データを格納
    response.data = csv_file.getvalue()
    return response


# *** DBをリセット ***
@app.route("/reset")
def reset():
    # 指定ディレクトリ内のファイルを取得
    images = glob("static/images/*.png")
    # DB内を一括で削除
    Image.query.delete()
    # 新たにDBへ登録
    for image in images:
        col = Image(filename=image)
        db_session.add(col)
    db_session.commit()  # 更新をDBに反映
    return render_template("reset.html", images=images)

# *** 画像をアップロード ***
@app.route("/upload")
def upload():
    return render_template("upload.html")

# *** 画像を識別 ***
@app.route("/classifier")
def classifier():
    img_path = "static/images/image01.png"
    result = img2category(img_path)
    return render_template("classifier.html", img=img_path, result=result)

# *** 画像を受ける ***
@app.route("/image", methods=['POST'])
def image():
    # print(request.headers['Content-Type'])
    # if request.headers['Content-Type'] != 'application/json':
    #     print(request.headers['Content-Type'])
    #     return jsonify(res='error'), 400
    # content = request.json
    for x in dir(request):
        print(x)
    print(request.form)
    # print(jsonify(request.json), file=sys.stderr)
    return jsonify(res='ok')



if __name__ == "__main__":
    app.run()
