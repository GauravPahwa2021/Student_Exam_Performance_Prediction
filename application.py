from flask import Flask,render_template,request,jsonify
from flask_cors import CORS,cross_origin

from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.logger import logging

application = Flask(__name__)
app = application

# Route for a home page
@app.route('/')
@cross_origin()
def home_page():
    return render_template('home.html')

@app.route('/predict',methods = ['GET','POST'])
@cross_origin()
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = float(request.form.get('writing_score')),
            writing_score = float(request.form.get('reading_score'))
        )

        predict_df = data.get_data_as_data_frame()
        print(predict_df)

        logging.info("Before Pediction")
        predict_pipeline = PredictPipeline()

        logging.info("Mid Prediction")
        results = predict_pipeline.predict(predict_df)

        logging.info("After Prediction")
        return render_template('home.html',results=results[0])
    
@app.route('/predictAPI',methods=['POST'])
@cross_origin()
def predict_api():
    if request.method=='POST':
        data = CustomData(
            gender = request.json['gender'],
            race_ethnicity = request.json['ethnicity'],
            parental_level_of_education = request.json['parental_level_of_education'],
            lunch = request.json['lunch'],
            test_preparation_course = request.json['test_preparation_course'],
            reading_score = float(request.json['writing_score']),
            writing_score = float(request.json['reading_score'])
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        dct = {'math_score':round(results[0],2)}
        return jsonify(dct)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000)  