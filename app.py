import os
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from functions import analyze_image

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            dominant_color = analyze_image(filepath)
            color_bgr = tuple(map(int, dominant_color))
            color_hex = "#{:02x}{:02x}{:02x}".format(color_bgr[2], color_bgr[1], color_bgr[0])
            
            return f"Dominant color in the facial region: BGR {color_bgr}, HEX {color_hex}"
        except ValueError as e:
            return str(e)

if __name__ == "__main__":
    app.run(debug=True)
