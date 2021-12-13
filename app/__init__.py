from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import app.preprocessing as pre
import app.regression_model as reg
import app.ann_model as ann
import app.RF_model as rf

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello():
    return "Hello!"

@app.route("/predict", methods=["POST"])
def postInput():
    #取得前端輸入的數值
    insertValues = request.get_json()
    x1 = insertValues["District"]
    x2 = insertValues["Object"]
    x3 = insertValues["Structure"]
    x4 = insertValues["Floor"]
    x5 = insertValues["Building_Height"]
    x6 = insertValues["Square"]
    x7 = insertValues["furniture"]
    x8 = insertValues["lat"]
    x9 = insertValues["lnt"]
    rent_data = [x1,x2,x3,x4,x5,x6,x7,x8,x9]

    result1 = reg.predict(rent_data)
    result2 = ann.predict(rent_data)
    result3 = rf.predict(rent_data)
    result = (result1 + result2) * 0,3 + result3 * 0.4

    return jsonify({"return" : result})

if __name__ == "__main__":
    from waitress import serve
    app.run(host="0.0.0.0", port=3000, debug=True)