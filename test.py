import cv2
import numpy as np
from deepface.basemodels import ArcFace
from deepface.commons import functions
from deepface.basemodels.retinaface.detector import RetinaFace
from tensorflow.keras.preprocessing import image
import os

import time
from elasticsearch import Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])


# ---------- read Video --------------------------
# cap = cv2.VideoCapture(0) #read from web camera
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
frame_width = 1920
frame_height = 1080
resolution = (frame_width, frame_height)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, resolution)

# ---------- call class retina face detector --------------------------
print("load retina face done!!")
face_detector = RetinaFace(gpu_id=0)
print("load ArcFace Models done!!")
arcface_model = ArcFace.loadModel()


def getAllDocNumber():
    res = es.count(index='face_recognition')
    res = res["count"]
    return res

def createElasticsearchIndex():
    mapping = {
        "mappings": {
            "properties": {
                "title_vector": {
                    "type": "dense_vector",
                    "dims": 512
                },
                "title_name": {"type": "keyword"}
            }
        }
    }
    es.indices.create(index="face_recognition", body=mapping)
    #Use Only one time for create index in elasticsearch

def search_face(target_embeddimg):
    query = {
        "size": 1,
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    # "source": "cosineSimilarity(params.queryVector, 'title_vector') + 1.0", #cosine distance
                    "source": "1 / (1 + l2norm(params.queryVector, 'title_vector'))", # euclidean distance
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

def addFaceToDatabaseWithImage():
    files = []
    for r, d, f in os.walk("./tests/dataset/"):
        for file in f:
            if file.endswith('.jpg'):
                files.append(r +'/'+ file)

    index = getAllDocNumber()
    for img_path in files:
        file_name = img_path.split('/')[3]
        img = functions.preprocess_face(img_path, target_size=(112, 112), detector_backend="retina",
                                        enforce_detection=False)

        embedding = arcface_model.predict(img)
        embedding = embedding[0]
        print(img_path)
        doc = {"title_name": file_name, "title_vector": embedding}
        index = index + 1
        es.create("face_recognition", id=index, body=doc)

def addFaceToDatabaseWithCamera():
    name = input('What is you name? : ')
    print("Pass Q for brake add face process")
    while True:
        isSuccess, frame = cap.read()
        preprocess_video(frame,name)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Terminate by user")
            break
    print("ADD FACE DONE !!")

def addFaceToDatabaseWithVideo(pathVideo):
    name = input('What is you name? : ')
    print("Pass Q for brake add face process")
    cap = cv2.VideoCapture(pathVideo)
    while True:
        try:
            isSuccess, frame = cap.read()
            preprocess_video(frame,name)
        except:
            pass
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Terminate by user")
            break
    print("ADD FACE DONE !!")

def preprocess_video(frame, name):
    index = getAllDocNumber()
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        img = functions.preprocess_face(frame, target_size=(112, 112), detector_backend="retina",
                                        enforce_detection=False)
        embedding = arcface_model.predict(img)
        embedding = embedding[0]
        doc = {"title_name": name, "title_vector": embedding}
        index = index + 1
        es.create("face_recognition", id=index, body=doc)
        print("ADD FACE !!")
    except:
        pass
    cv2.imshow("", frame)

def preprocess_face(img):
    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    faces = face_detector(img)
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

def drawingBBox(frame,res_obj):
    for img, box, score in res_obj:
        embedding = arcface_model.predict(img)
        embedding = embedding[0]
        score, source = search_face(embedding)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=1)
        if score > 0.25:
            cv2.putText(frame, 'UNKNOW', (box[0], box[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (255, 255, 255), 2)
            continue
        cv2.putText(frame, source, (box[0], box[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                    (255, 255, 255), 2)

def streamOnCammera():
    while True:
        isSuccess, frame = cap.read()
        try:
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t0 = time.time()
            res_obj = preprocess_face(frame)
            drawingBBox(frame, res_obj)
        except:
            pass
        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Terminate by user")
            break
        t1 = time.time()
        print("frame")
        print(f'took {round(t1 - t0, 3)} to process')

def streamOnVideo(pathVideo):
    cap = cv2.VideoCapture(pathVideo)
    while True:
        isSuccess, frame = cap.read()
        try:
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t0 = time.time()
            res_obj = preprocess_face(frame)
            drawingBBox(frame, res_obj)
        except:
            pass
        out.write(frame)
        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Terminate by user")
            break
        t1 = time.time()
        print("frame")
        print(f'took {round(t1 - t0, 3)} to process')

def processOnImage(img):
    res_obj = preprocess_face(img)
    for img, box, score in res_obj:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=1)


if __name__ == "__main__":
    # img = cv2.imread('./tests/dataset/lisatest.jpg')
    # processOnImage(img)
    # createElasticsearchIndex()
    # addFaceToDatabaseWithImage()
    # addFaceToDatabaseWithCamera()
    # addFaceToDatabaseWithVideo('./tests/video/toei.MOV')
    # streamOnCammera()
    streamOnVideo('./tests/video/testVideo1.mp4')
