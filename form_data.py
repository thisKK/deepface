from camera import VideoCamera
from flask import Flask, render_template, request, jsonify, Response
import base64,cv2
import time
from tqdm import tqdm

from deepface.basemodels.retinaface.detector import RetinaFace
import numpy as np
from deepface.commons import functions
from tensorflow.keras.preprocessing import image
from elasticsearch import Elasticsearch

from deepface import DeepFace
import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])
app = Flask(__name__)

# ------------------------------

tic = time.time()

print("Loading Face Recognition Models...")

pbar = tqdm(range(0, 6), desc='Loading Face Recognition Models...')

for index in pbar:
    if index == 0:
        print(index)
        # pbar.set_description("Loading VGG-Face")
        # vggface_model = DeepFace.build_model("VGG-Face")
    elif index == 1:
        print(index)
        # pbar.set_description("Loading OpenFace")
        # openface_model = DeepFace.build_model("OpenFace")
    elif index == 2:
        print(index)
        # pbar.set_description("Loading Google FaceNet")
        # facenet_model = DeepFace.build_model("Facenet")
    elif index == 3:
        print(index)
        # pbar.set_description("Loading Facebook DeepFace")
        # deepface_model = DeepFace.build_model("DeepFace")
    elif index == 4:
        print(index)
        # pbar.set_description("Loading DeepID DeepFace")
        # deepid_model = DeepFace.build_model("DeepID")
    elif index == 5:
        pbar.set_description("Loading ArcFace DeepFace")
        arcface_model = DeepFace.build_model("ArcFace")
toc = time.time()

print("Face recognition models are built in ", toc - tic, " seconds")

tic = time.time()
print("Loading Face Detector Models...")
pbar = tqdm(range(0, 1), desc='Loading Face Detector Models...')
for index in pbar:
    if index == 0:
        detector = RetinaFace(gpu_id=0)

toc = time.time()
print("Face Detector models are built in ", toc - tic, " seconds")

def gen(camera):
    while True:
        frame = camera.get_frame()
        res_obj = preprocess_face(frame)
        for img, box, score in res_obj:
            embedding = arcface_model.predict(img)
            embedding = embedding[0]
            score, source = search_face(embedding)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=1)
            cv2.putText(frame, source + str(score), (box[0], box[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (255, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        data = []
        data.append(jpeg.tobytes())
        frame = data[0]
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def search_face(target_embeddimg):
    es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
    query = {
        "size": 1,
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    # "source": "cosineSimilarity(params.queryVector, 'title_vector') + 1.0",
                    "source": "1 / (1 + l2norm(params.queryVector, 'title_vector'))",
                    # euclidean distance
                    "params": {
                        "queryVector": list(target_embeddimg)
                    }
                }
            }
        }}

    res = es.search(index='face_recognition', body=query)

    for i in res["hits"]["hits"]:
        score = i["_score"]
        source = i["_source"]["title_name"]
        print(score, source)
        print("-------------------------------------------")
        return score, source

def preprocess_face(img):
    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    faces = detector(img)
    res_obj = []
    for box, landmarks, score in faces:
        result = ()
        score.astype(np.float)
        box = box.astype(np.int)
        if score < 0.4:
            continue
        cropped = img[box[1]-10:box[3]+10, box[0]-10:box[2]+10]
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            img_alignment = functions.alignment_procedure(cropped, left_eye, right_eye)
        else:
            img_alignment = cropped

        cropped = cv2.resize(img_alignment, (112, 112))
        # cv2.imwrite(f"result_out.jpg", cropped)
        img_pixels = image.img_to_array(cropped)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255  # normalize input in [0, 1]

        result = (img_pixels, box, score)
        res_obj.append(result)
    return res_obj

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<h1>Hello, world!</h1>'

if __name__=="__main__":
    app.run(debug=True)#,host="192.168.43.161")
