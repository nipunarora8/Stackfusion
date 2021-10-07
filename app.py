from flask import Flask, redirect, url_for, request, render_template,send_from_directory
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import os,cv2
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from model_run.detect import main
app = Flask(__name__)

@app.route('/predict', methods=['POST','GET'])
def predict():
    
    if request.method == 'POST':
        
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        file_path = file_path.replace('\\','/')
        f.save(file_path)

        image, plate, result = main(file_path)
        
        plate_file_path = file_path.split('.')
        plate_file_path[-1] = '_plate.'+plate_file_path[-1]
        plate_file_path = "".join(plate_file_path)

        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        cv2.imwrite(file_path, image)
        plate = cv2.cvtColor(np.array(plate), cv2.COLOR_BGR2RGB)
        cv2.imwrite(plate_file_path, plate)
        
        return render_template('pred.html',number=result.upper(),file_name=str(f.filename), plate_file_name=plate_file_path.split('/')[-1])
	
@app.route('/uploads/<filename>')
def upload_img(filename):
    return send_from_directory("uploads", filename)

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)