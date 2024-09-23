from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and scaler
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
scaler = StandardScaler()  # You may need to load your scaler if it was saved separately

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  # Get data from the HTML form
  X = int(request.form['x'])
  Y = int(request.form['y'])
  FFMC = float(request.form['ffmc'])
  DMC = float(request.form['dmc'])
  DC = float(request.form['dc'])
  ISI = float(request.form['isi'])
  temp = float(request.form['temperature'])
  RH = float(request.form['humidity'])
  wind = float(request.form['wind'])
  rain = float(request.form['rain'])

  # Create a DataFrame from the form data
  new_data = pd.DataFrame([[X, Y, FFMC, DMC, DC, ISI, temp, RH, wind, rain]], 
                         columns=['X', 'Y', 'Fine Fuel Moisture Code', ' Duff Moisture Code', 'Drought Code', 'Initial Spread Index', 'Temperature ', 'Relative humidity', 'Wind', 'Rain'])
  
  scaler.fit_transform(new_data)
  # Scale the new data
  new_scaled = scaler.transform(new_data)
  
  # Make a prediction using the loaded model
  prediction = rf_model.predict(new_scaled)[0]
  
  if (prediction == 0):
    return render_template('index.html', prediction_text='Low'.format(prediction))
    
  elif (1>=prediction>0):
    return render_template('index.html', prediction_text= 'High'.format(prediction_text=prediction))
  
  else:
    return render_template('index.html', prediction_text= 'Very High'.format(prediction_text=prediction))

if __name__ == '__main__':
  app.run(debug=True) 
