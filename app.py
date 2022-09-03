import numpy as np
from flask import Flask,render_template,request,Blueprint
import pandas as pd
from werkzeug.utils import secure_filename
import os
from data_handling.ewm import data_hd
import numpy as np
import json


app = Flask(__name__)


def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        # 如果不存在则创建目录
        os.makedirs(path)
        print(path + '创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + '目录已存在')
        return False

# 上传数据保存在upload文件夹中
mkpath = ".\\upload"
mkdir(mkpath)

app.config['UPLOAD_FOLDER'] = 'upload/'


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/upload', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        global f
        f = request.files["data_file"]
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        data = pd.read_excel(f, converters={'Scode': str})
        code = data.loc[1, "Scode"]
        data = data.to_dict(orient='records')  # 将数据构建成html需要的样式
        return render_template('data_table.html', data=data, code=code)
    else:
        return render_template('upload.html')


@app.route('/display')
def radar_show():
    data = pd.read_excel(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)), converters={'Scode': str})
    code = data.loc[1, "Scode"]
    data_show = data_hd(data).iloc[:, 1:]
    data_list = []
    for i in data_show.columns:
        data_list.append(np.array(data_show[i]).tolist())
    data_show = np.array(data_show).tolist()  # 将数据构建成html需要的样式

    return render_template('display.html', data_list=data_list, data_show=data_show, code=code)


if __name__ == '__main__':
    app.run(debug=True)

