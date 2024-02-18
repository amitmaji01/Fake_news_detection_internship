from flask import Flask,request,render_template,jsonify
import numpy as numpy
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

application=Flask(__name__)
app=application
#import model and scaler 
model=pickle.load(open('model/model_mnb.pkl','rb'))
vector=pickle.load(open('model/vectorization.pkl','rb'))

#route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
     if request.method=='POST':
        news=str(request.form.get('news'))

        result=model.predict(vector.transform([news]))
        if(result==1):
            result='True news'
        else:
            result='Fake news'


        return render_template('index.html',result=result)
     
     else:
        return render_template('index.html')







if __name__=="__main__":
    app.run(host="0.0.0.0")

