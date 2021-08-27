import pickle

infile = open('./static/model','rb')
sm_model = pickle.load(infile)
infile.close()

infile = open('./static/scaler','rb')
scaler = pickle.load(infile)
infile.close()

# Code for inference
temperature = 13.05
windSpeed = 14.27

cols = scaler.transform([[temperature,18,80,windSpeed,300,500,800]])[0]

temperature, windSpeed = cols[0], cols[3]

cols[6] = predicted = sm_model.predict([[1,1,temperature,windSpeed,0,0,0,0,1,1,0,0]])

predicted = scaler.inverse_transform([cols])[0][6]

print(predicted)