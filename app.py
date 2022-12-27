from flask import Flask, request, render_template
import numpy as np
import pickle

#Create an app  object using the flask class
app = Flask(__name__, template_folder='template', static_folder='image')



#use route() decorator to tell flask what url should trigger action
@app.route('/')
def home():
    return render_template('index.html')

def predict(input_data):
   #load the trained model
    model = pickle.load(open('model/model.pkl', 'rb'))
   # Make a prediction using the model and input data
    prediction = model.predict(input_data)

    return prediction

@app.route('/predict', methods=['POST'])
def make_prediction():

    #int_features = [int(X) for X in request.form.values()]
    #features = [np.array(int_features)]
    input_data1= int(request.form['input_data1'])
    input_data2 = int(request.form['input_data2'])
    input_data3 = int(request.form['input_data3'])
    input_data = np.array([input_data1, input_data2, input_data3]).reshape(-1,3)
    prediction = predict(input_data)

    output = np.round(prediction)

    return render_template('index.html', prediction_text = 'House Price is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)