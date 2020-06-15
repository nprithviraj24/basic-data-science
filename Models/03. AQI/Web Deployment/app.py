from flask import Flask,flash,redirect,url_for
from flask import Flask, render_template, url_for, request, session, jsonify
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import StringField, PasswordField, BooleanField,DecimalField, SubmitField, SelectField
from wtforms.validators import DataRequired
import os
from sklearn.externals import joblib
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os.path



app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.secret_key = os.urandom(24)

parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir)); 


type_en = open(os.path.join(parent,'Type_encoder.pkl'), 'rb')
label1 = pickle.load(type_en) 
type_en.close()

location_en = open(os.path.join(parent,'Location_encoder.pkl'), 'rb')
label2 = pickle.load(location_en) 
location_en.close()


aqi_en = open(os.path.join(parent,'AQI_Range_encoder.pkl'), 'rb')
label3 = pickle.load(aqi_en) 
aqi_en.close()


model1 = joblib.load(os.path.join(parent,'AQI_model.pkl'))
model2 = joblib.load(os.path.join(parent,'AQI_Range_model.pkl'))




class LoginForm(FlaskForm):
    username = StringField('Username:', validators=[DataRequired()])
    password = PasswordField('Password:', validators=[DataRequired()])
    submit = SubmitField('Log In')

class MLdata(FlaskForm):
	location = SelectField('Location:', choices = [('ASANSOL', 'ASANSOL'), ('BARRACKPORE','BARRACKPORE'), ('BARUIPUR','BARUIPUR'),
													('CALCUTTA','CALCUTTA'),('DANKUNI','DANKUNI'), ('DURGAPUR','DURGAPUR'),
													('DURGAPUR (WB)', 'DURGAPUR (WB)'), ('HALDIA', 'HALDIA'), ('HOWRAH','HOWRAH'),
													('KALYANI', 'KALYANI'), ('kl','KOLKATA'), ('MALDAH','MALDAH'),
													('RANIGANJ', 'RANIGANJ'),('SANKRAIL','SANKRAIL'), ('SILIGURI','SILIGURI'),
													('so','SOUTH SUBURBAN'), ('ULUBERIA','ULUBERIA')])
	type1 = SelectField('Type:', choices = [('Industrial Area', 'Industrial Area'), ('RIRUO','RIRUO'), ('Residential', 'Residential')])
	so2 = DecimalField('so2:', validators=[DataRequired()])
	no2 = DecimalField('no2:', validators=[DataRequired()])
	rspm = DecimalField('rspm:', validators=[DataRequired()])
	spm = DecimalField('spm:', validators=[DataRequired()])
	pm2_5 = DecimalField('pm2_5:', validators=[DataRequired()])
	SOi = DecimalField('soi:', validators=[DataRequired()])
	Noi = DecimalField('Noi:', validators=[DataRequired()])
	RSPMi = DecimalField('RSPMi:', validators=[DataRequired()])
	SPMi = DecimalField('SPMi:', validators=[DataRequired()])
	PMi = DecimalField('PMi:', validators=[DataRequired()])
	submit = SubmitField('Predict')

'''
@app.route('/predict', methods=['GET','POST'])
def predict():
	print("Hello")
	if request.method == 'POST':  # if the file is submitted post request is send using ajax
		file = request.files['file']
	target = os.path.join(APP_ROOT, 'static/images/')
	#print("Hello World")
	# print(file)
	filename = secure_filename(file.filename) 
	session['img'] = filename
	destination = "/".join([target, filename]) 
	file.save(destination)  
	target = os.path.join(APP_ROOT, 'images/')
	print(target)
	img_src = session['img']
	model = load_model('../model.h5', compile=False)
	image = cv2.imread("static/images/{}".format(img_src))
	input_image = cv2.resize(image, (256, 256))
	input_image = input_image / 255.
	input_image = input_image[:,:,::-1]
	input_image = np.expand_dims(input_image, 0)
	result = model.predict(input_image)
	K.clear_session()
	if result[0][0] > result[0][1]:
		msg = "The patient's cancer is in benign stage with " + str(result[0][0] * 100) + " %."
	else:
		msg = "The patient's cancer is in malignant stage with " + str(result[0][1] * 100) + " %. Take Care!!"
	begin = str(result[0][0])
	malignant = str(result[0][1])
	#dic = {'Begin': begin, 'Malignant':malignant, 'out': msg}
	dic = {'Prediction': msg}
	return jsonify (dic)
'''

@app.route('/prediction',methods=['GET', 'POST'])
def prediction():
	form = MLdata()
	if request.method == 'POST':
		if form.validate_on_submit():
			location = str(form.location.data)
			type1 = str(form.type1.data)
			so2 = float(form.so2.data)
			no2 = float(form.no2.data)
			rspm = float(form.rspm.data)
			spm = float(form.spm.data)
			pm2_5 = float(form.pm2_5.data)
			SOi = float(form.SOi.data)
			Noi = float(form.Noi.data)
			RSPMi = float(form.RSPMi.data)
			SPMi = float(form.SPMi.data)
			PMi = float(form.PMi.data)
			#print(location)
			location = label2.transform([location])
			type1 = label1.transform([type1])

			location = int(location[0])
			type1 = int(type1[0])

			test_point = [{'location': location, 'type1': type1, 'so2': so2, 'no2': no2, 'rspm': rspm,
							'spm': spm, 'pm2_5': pm2_5, 'SOi': SOi, 'NOi': Noi,'RSPMi': RSPMi,'SPMi': SPMi,
  							'PMi': PMi}]
			df = pd.DataFrame(test_point) 
			#print(df)
			sampleDataFeatures = np.asarray(df)

			#print(sampleDataFeatures)
			a = StandardScaler()
			#print("****************")
			sampleDataFeatures = a.fit_transform(sampleDataFeatures)

			prediction2 = model2.predict(sampleDataFeatures)
			prediction2 = label3.inverse_transform(prediction2)

			prediction1 = model1.predict(sampleDataFeatures)

			print(prediction1[0], prediction2[0])
			data1 = []
			data2 = []
			data = [{'AQI Value': prediction1[0], 'AQI Range': prediction2[0]}]
			data1.append(prediction1[0])
			data2.append(prediction2[0])
			#print(form.so2.data)
			#print(form.no2.data)
			#print(form.pm2_5.data)
	#return redirect(url_for('main'))
	return render_template('Result.html', title='Result Page', form = form, data = data)


@app.route('/main',methods=['GET', 'POST'])
def main():
	#print("**********************************")
	form = MLdata()
	'''
	if request.method == 'POST':
		if form.validate_on_submit(): 
			print(form.so2.data)
			print(form.no2.data)
			#return redirect(url_for('main'))
	'''


	return render_template('main_page.html', title='Main Page', form = form)


@app.route('/',methods=['GET', 'POST'])
def home():
	form = LoginForm()
	if request.method == 'POST':
		if form.validate_on_submit():  # POST request with valid input?
			# Verify username and passwd
			if (form.username.data == 'air' and form.password.data == 'quality'):
				return redirect(url_for('main'))
			else:
				# Using Flask's flash to output an error message
				flash('Invalid username or password')

	return render_template('login.html', title='Sign In', form=form)

if __name__ == '__main__':  # Script executed directly
    app.run(debug=True)  # Launch built-in web server and run this Flask webapp