from flask import Flask,request,render_template,jsonify
from keras.models import model_from_json
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow as tf
from keras import backend as K
K.clear_session()

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


app = Flask(__name__)

@app.route('/')
def index():
	return render_template('subli.html')



@app.route('/predict',methods= ['POST'])
def process():
  arquivo = open('model_crio1.json','r')
  estrutura_rede = arquivo.read()
  model = model_from_json(estrutura_rede)
  model.load_weights('model_crio1.h5')
  
  dado1 = request.form['dado1']
  dado2 = request.form['dado2']
  
  dado1 = float(dado1)
  dado2 = float(dado2)
  
                    
  previviabilidade = [[dado2/100], [dado1/100],[0],[0]]
                    
  previviabilidade = np.asmatrix(previviabilidade)    
                    
  previviabilidade = previviabilidade.T
  
  previviabilidade = model.predict(previviabilidade)
  
  Viabilidade = previviabilidade*100

 
 
  
  #output = dado1 + dado2
  
  output = str(Viabilidade)
  
  if dado1 and dado2:
   return jsonify({'output':'Predicted Viability: ' + output+'  %'})
  #return jsonify({'error' : 'Missing data!'})

if __name__ == '__main__':
	app.run(debug=True, threaded = False)