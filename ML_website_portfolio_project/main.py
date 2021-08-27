from flask import Flask, render_template, request
app = Flask(__name__)

import pickle

infile = open('./static/bike_sharing_model','rb')
bike_sharing_model = pickle.load(infile)
infile.close()

infile = open('./static/bike_sharing_scaler','rb')
bike_sharing_scaler = pickle.load(infile)
infile.close()


infile = open('./static/heart_attack_model','rb')
heart_attack_model = pickle.load(infile)
infile.close()

infile = open('./static/heart_attack_scaler','rb')
heart_attack_scaler = pickle.load(infile)
infile.close()


@app.route('/')
def home():
    return render_template('index.html',title='Home Page')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/bike_sharing', methods=["GET", "POST"])
def bike_sharing_project():
    if request.method == "POST":
        print(request.form)
        dict = request.form

        year = int(dict['year'])
        temperature = float(dict['temperature'])
        windSpeed = float(dict['windspeed'])
        Sep = int(dict['Sep'])
        Sun = int(dict['Sun'])

        if int(dict['Season']) == 1:
            Spring = 1
            Summer = 0
            Winter = 0
        elif int(dict['Season']) == 2:
            Spring = 0
            Summer = 1
            Winter = 0
        elif int(dict['Season']) == 3:
            Spring = 0
            Summer = 0
            Winter = 1
        elif int(dict['Season']) == 4:
            Spring = 0
            Summer = 0
            Winter = 0

        if Sep == 1:
            Spring = 0
            Summer = 0
            Winter = 0

        if int(dict['Weather']) == 1:
            Cloudy = 1
            Light_precip = 0
        elif int(dict['Weather']) == 2:
            Cloudy = 0
            Light_precip = 1
        elif int(dict['Weather']) == 3:
            Cloudy = 0
            Light_precip = 0

        holiday = int(dict['Day'])

        # Code for inference
        cols = bike_sharing_scaler.transform([[temperature,18,80,windSpeed,300,500,800]])[0]

        temperature, windSpeed = cols[0], cols[3]

        cols[6] = predicted = bike_sharing_model.predict([[1,year,temperature,windSpeed,Sep,Sun,Spring,Summer,Winter,Cloudy,Light_precip,holiday]])

        predicted = bike_sharing_scaler.inverse_transform([cols])[0][6]

        print(predicted)

        return render_template('show.html', param=round(predicted))
    return render_template('/projects/bike_sharing.html')



@app.route('/heart_attack', methods=["GET", "POST"])
def heart_attack_project():
    if request.method == "POST":
        print(request.form)
        dict = request.form

        sex = int(dict['sex'])
        thalach = float(dict['thalach'])
        exang = int(dict['exang'])
        oldpeak = float(dict['oldpeak'])
        ca = int(dict['ca'])

        # Code for inference
        cols = heart_attack_scaler.transform([[50, 145, 250, thalach, oldpeak, ca]])[0]

        thalach, oldpeak, ca = cols[3], cols[4], cols[5]

        predicted = heart_attack_model.predict([[1,sex,thalach,exang,oldpeak,ca]])[0]

        if predicted > 0.5:
            result = 'It is a heart attack'
        else:
            result = 'It is not a heart attack'

        print(predicted)

        return render_template('show.html', param=result)
    return render_template('/projects/heart_attack.html')


if __name__ == "__main__":
    app.run(debug=True)
