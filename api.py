import jsonpickle
from _dlib_pybind11.image_dataset_metadata import image
from flask import Flask, request, jsonify, Response , json
from flask_cors import CORS

import base64, io
import argparse
import uuid
import time
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from base64 import encodebytes

from camera import VideoCamera
from deepface.basemodels.retinaface.detector import RetinaFace
from deepface.commons import functions
from elasticsearch import Elasticsearch
from deepface import DeepFace

from tensorflow.keras.preprocessing import image
import tensorflow as tf

tf_version = int(tf.__version__.split(".")[0])

# ------------------------------

app = Flask(__name__)

CORS(app, resources={r'/*': {'origins': '*'}})

# ------------------------------

tic = time.time()

print("Loading Face Recognition Models...")

pbar = tqdm(range(0, 6), desc='Loading Face Recognition Models...')

for index in pbar:

    if index == 0:
        pbar.set_description("Loading VGG-Face")
        vggface_model = DeepFace.build_model("VGG-Face")
    elif index == 1:
        pbar.set_description("Loading OpenFace")
        openface_model = DeepFace.build_model("OpenFace")
    elif index == 2:
        pbar.set_description("Loading Google FaceNet")
        facenet_model = DeepFace.build_model("Facenet")
    elif index == 3:
        pbar.set_description("Loading Facebook DeepFace")
        deepface_model = DeepFace.build_model("DeepFace")
    elif index == 4:
        pbar.set_description("Loading DeepID DeepFace")
        deepid_model = DeepFace.build_model("DeepID")
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

if tf_version == 1:
    graph = tf.get_default_graph()


# ------------------------------


# Service API Interface

@app.route('/')
def index():
    return '<h1>Hello, world!</h1>'

@app.route('/detection', methods=['POST'])
def detection():
    req = request.get_json()
    resp_obj = jsonify({'success': False})
    resp_obj = detectWrapper(req)
    return resp_obj, 200


@app.route('/verify', methods=['POST'])
def verify():
    global graph

    tic = time.time()
    req = request.get_json()
    trx_id = uuid.uuid4()

    resp_obj = jsonify({'success': False})

    if tf_version == 1:
        with graph.as_default():
            resp_obj = verifyWrapper(req, trx_id)
    elif tf_version == 2:
        resp_obj = verifyWrapper(req, trx_id)

    # --------------------------

    toc = time.time()

    resp_obj["trx_id"] = trx_id
    resp_obj["seconds"] = toc - tic

    return resp_obj, 200


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


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


def search_face(target_embedding):
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
                        "queryVector": list(target_embedding)
                    }
                }
            }
        }}

    res = es.search(index='face_recognition', body=query)

    for i in res["hits"]["hits"]:
        score = i["_score"]
    source = i["_source"]["title_name"]
    return score, source


# Take in base64 string and return cv image
def stringToRGB(base64_string):
    print(base64_string)
    imgdata = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


def detectface(img):
    faces = detector(img)
    res_obj = []
    count = 0

    for box, landmarks, score in faces:
        count = count+1
        result = ()
        score.astype(float)
        box = box.astype(int)
        landmarks = landmarks.astype(int)
        if score < 0.4:
            continue
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 200), thickness=1)

        box = box.tolist()
        landmarks = landmarks.tolist()
        score = score.astype(str)
        # result = (box, score)
        resp_obj = {
             "faceLandmarks": landmarks
            , "faceRectangle": box
            , "model": "retina"
            , "confidence": score
            , "faceNumber": count
        }
        res_obj.append(resp_obj)
    # ret, jpeg = cv2.imencode('.jpg', img)
    # response_pickled = jsonpickle.encode(jpeg)
    # res_obj.append(response_pickled)

    return res_obj

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
        cropped = img[box[1] - 10:box[3] + 10, box[0] - 10:box[2] + 10]
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            img_alignment = functions.alignment_procedure(cropped, left_eye, right_eye)
        else:
            img_alignment = cropped
        cropped = cv2.resize(img_alignment, (112, 112))

        img_pixels = image.img_to_array(cropped)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255  # normalize input in [0, 1]

        result = (img_pixels, box, score)
        res_obj.append(result)

    return res_obj


def detectWrapper(req):
    resp_obj = jsonify({'success': False})
    model_name = "retina"
    if "model_name" in list(req.keys()):
        model_name = req["model_name"]

    instances = []
    if "img" in list(req.keys()):
        raw_content = req["img"]  # list

        # for item in raw_content:  # item is in type of dict
        #     instance = []
        #     img = item["img"]
        #
        #     validate_img = False
        #     if len(img) > 11 and img[0:11] == "data:image/":
        #         validate_img = True
        #
        #     if validate_img != True:
        #         return jsonify({'success': False, 'error': 'you must pass both img as base64 encoded string'}), 205

        img = functions.loadBase64Img(raw_content)
        faces = detectface(img)
        resp_obj = json.dumps(faces)

            # instance.append(img)
            # instances.append(instance)

    # --------------------------
    # if len(instances) == 0:
    #     return jsonify({'success': False, 'error': 'you must pass at least one img object in your request'}), 205

    # for img in instances:
    #     if model_name == "retina":
    #         img = functions.loadBase64Img(img)
    #         faces = detectface(img)
    #         resp_obj = json.dumps(faces)

    return resp_obj


def verifyWrapper(req, trx_id=0):
    resp_obj = jsonify({'success': False})

    model_name = "VGG-Face"
    distance_metric = "cosine"
    if "model_name" in list(req.keys()):
        model_name = req["model_name"]

    if "distance_metric" in list(req.keys()):
        distance_metric = req["distance_metric"]

    # ----------------------

    instances = []
    if "img" in list(req.keys()):
        raw_content = req["img"]  # list

        for item in raw_content:  # item is in type of dict
            instance = []
            img1 = item["img1"]
            img2 = item["img2"]

            validate_img1 = False
            if len(img1) > 11 and img1[0:11] == "data:image/":
                validate_img1 = True

            validate_img2 = False
            if len(img2) > 11 and img2[0:11] == "data:image/":
                validate_img2 = True

            if validate_img1 != True or validate_img2 != True:
                return jsonify(
                    {'success': False, 'error': 'you must pass both img1 and img2 as base64 encoded string'}), 205

            instance.append(img1)
            instance.append(img2)
            instances.append(instance)

    # --------------------------

    if len(instances) == 0:
        return jsonify({'success': False, 'error': 'you must pass at least one img object in your request'}), 205

    print("Input request of ", trx_id, " has ", len(instances), " pairs to verify")

    # --------------------------

    if model_name == "VGG-Face":
        resp_obj = DeepFace.verify(instances, model_name=model_name, distance_metric=distance_metric,
                                   model=vggface_model, detector_backend="retina")
    elif model_name == "Facenet":
        resp_obj = DeepFace.verify(instances, model_name=model_name, distance_metric=distance_metric,
                                   model=facenet_model, detector_backend="retina")
    elif model_name == "OpenFace":
        resp_obj = DeepFace.verify(instances, model_name=model_name, distance_metric=distance_metric,
                                   model=openface_model, detector_backend="retina")
    elif model_name == "DeepFace":
        resp_obj = DeepFace.verify(instances, model_name=model_name, distance_metric=distance_metric,
                                   model=deepface_model, detector_backend="retina")
    elif model_name == "DeepID":
        resp_obj = DeepFace.verify(instances, model_name=model_name, distance_metric=distance_metric,
                                   model=deepid_model, detector_backend="retina")
    elif model_name == "ArcFace":
        resp_obj = DeepFace.verify(instances, model_name=model_name, distance_metric=distance_metric,
                                   model=arcface_model, detector_backend="retina")
    # elif model_name == "Ensemble":
    # 	models =  {}
    # 	models["VGG-Face"] = vggface_model
    # 	models["Facenet"] = facenet_model
    # 	models["OpenFace"] = openface_model
    # 	models["DeepFace"] = deepface_model
    # 	resp_obj = DeepFace.verify(instances, model_name = model_name, model = models)
    else:
        resp_obj = jsonify(
            {'success': False, 'error': 'You must pass a valid model name. You passed %s' % model_name}), 205

    return resp_obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=5000,
        help='Port of serving api')
    args = parser.parse_args()
    app.run(host='127.0.0.1', port=args.port)
