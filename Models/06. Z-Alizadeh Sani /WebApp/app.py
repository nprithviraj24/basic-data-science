from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np
import pickle
import os

def load_pickle(name):
    models_path = os.path.abspath(os.path.join(os.getcwd(), 'models'))
    this_path = os.path.join( models_path ,name)
    with open(this_path, 'rb') as fid:
        model = pickle.load(fid)
    return model

# app = Flask('Swaroop')
app = Flask(__name__)
app.debug = True
# model1, model2, model3, model4, model5 = load_models()
model4 = load_pickle('ensemble_soft.pkl')
sc = load_pickle('standard_scaler.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        age = request.form['age']
        weight = request.form['weight']
        length = request.form['length']
        bmi = request.form['bmi']
        bp = request.form['bp']
        htn = request.form['htn']
        typical_chest_pain = request.form['typical chest pain']
        atypical = request.form['atypical']
        nonanginal = request.form['nonanginal']
        fbs = request.form['fbs']
        cr = request.form['cr']
        tg = request.form['tg']
        ldl = request.form['ldl']
        hdl = request.form['hdl']
        bun = request.form['bun']
        esr = request.form['esr']
        hb = request.form['hb']
        k = request.form['k']
        wbc = request.form['wbc']
        lymph = request.form['lymph']
        neut = request.form['neut']
        plt = request.form['plt']
        ef_tte = request.form['ef-tte']
        region_rwma = request.form['region-rwma']

        inputData = [[age, weight, length, bmi, htn, bp, typical_chest_pain, atypical, nonanginal, fbs, cr, tg, ldl, hdl, bun, esr, hb, k, wbc, lymph, neut, plt, ef_tte, region_rwma]]


        # # normal1 = [[66,67,158,26.83864765,1,100,0,0,1,78,1.2,63,55,27,30,76,12.1,4.4,13000,18,72,742,55,0]]
        # # inputData = normal1

        # # normal2 = [[41,68,169,23.80869017,0,130,0,0,0,65,1.1,69,130,22,14,13,11.4,4.6,7800,48,50,199,35,0]]
        # # inputData = normal2

        # # cad1 = [[59,81,167,29.04370899,0,120,1,0,0,218,1.1,130,110,45,11,19,13.3,4.7,12100,30,70,280,30,0]]
        # # inputData = cad1

        # cad2 = [[64,70,175,22.85714286,0,100,0,1,0,112,0.8,147,82,25,20,67,12.3,3.7,11300,23,70,266,50,0]]
        # inputData = cad2


        inputData = np.asarray(inputData, dtype='float64')
        inputData = sc.transform(inputData)

        output = str(model4.predict(inputData)[0])
        app.logger.warning('Output: ' + str(output))
        return redirect(url_for('show', output=output))

    return render_template('index.html')

@app.route('/show', methods=['GET'])
def show():
    output = request.args['output']
    app.logger.warning('Output at show endpoint: ' + str(output))
    return render_template('show.html', output=float(output))
