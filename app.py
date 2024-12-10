from flask import Flask,request,render_template,jsonify
import numpy as numpy
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

application=Flask(__name__)
app=application

model=pickle.load(open('model/model_mnb.pkl','rb'))
vector=pickle.load(open('model/vectorization.pkl','rb'))


@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/predictdata',methods=['GET','POST'])
# def predict_datapoint():
#      if request.is_json:
#         data = request.get_json()
#         news = data.get('news', '')

#         result=model.predict(vector.transform([news]))
#         if(result==1):
#             result='1'
#         else:
#             result='0'


#         return jsonify({result:"result"}), 200
     
#      else:
#         return jsonify({result:"-1"}), 301

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    if not request.is_json:
        return jsonify({"error": "Invalid request format. JSON expected."}), 400

    data = request.get_json()
    news = data.get('news', None)  


    if not news:
        return jsonify({"error": "The 'news' field is required and cannot be empty."}), 400

    try:
        
        result = model.predict(vector.transform([news]))
        prediction = "1" if result[0] == 1 else "0"

        return jsonify({"result": prediction}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__=="__main__":
    app.run(host="0.0.0.0")

