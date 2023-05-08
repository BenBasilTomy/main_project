from flask import Flask, request, render_template, jsonify
from datetime import datetime
import os
import ffmpeg
from yolov5.my_detect import run

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/',methods=['GET'])
def home():
    return render_template("index.html")



@app.route('/upload', methods=['POST'])
def upload_file():
    if request.files['file'].filename == '':
        response = {"status": 200, "msg": "No File Uploaded"}
        return jsonify(response)
    
    filename = str(datetime.now().microsecond) + '.mp4'
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    file.save(filepath)


    


    

    
    dir, labels, sums= run(weights='./models/best.pt',source=filepath,  view_img=False, line_thickness=1,
               project='./Detections', max_det=30)
    
    outfile = str(dir) + '/'+filename

    predicted_class = []
    classes =['Elephant','Wild Boar','Leopard']
    for i in range(len(sums)):
        if sums[i] != 0:
            print(classes[i])
            if classes[i] == 'Wild Boar':
                img_url='D:/website templates/breed/static/img/wildboar_bg.jpg'
                predicted_class.append(classes[i])
    
        
    print(predicted_class)
    print(labels)
    bounding_boxes = "video.mp4"
    response = {"status": 200, "msg": predicted_class,"video_preview": bounding_boxes, "img_url":img_url}
    return jsonify(response)
    return render_template("result.html",
                           file=outfile,
                           labels=labels, sums=sums)



if __name__ =='__main__':
    app.run(port=5000, debug=True)