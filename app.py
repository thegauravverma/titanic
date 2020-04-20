import numpy as np
from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction==0:
        value="CONGRATULATIONS,YOU ARE DEAD"
    else:
        value="CONGRATULATIONS,YOU ARE ALIVE"    
    return render_template('index.html', prediction_text=value)


if __name__ == "__main__":
    app.run(debug=True)