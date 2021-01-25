import cv2
import numpy as np
from deepface.basemodels import ArcFace
from deepface.commons import functions
from deepface.basemodels.retinaface.detector import RetinaFace
from tensorflow.keras.preprocessing import image

import time
from elasticsearch import Elasticsearch


# ---------- read Video --------------------------
# cap = cv2.VideoCapture(0) #read from web camera
cap = cv2.VideoCapture(0)

# # ---------- write out Video result -----------------
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# out = cv2.VideoWriter('output.mp4', fourcc, 15.0, (1920, 1080))
# cap.set(3, 1920)
# cap.set(4, 1080)

def addFaceToElasticSearch():
    print("hello")


def search_face(target_embeddimg):
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

def preprocess_face(img,target_size):
    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_detector(img)
    res_obj = []
    for box, landmarks, score in faces:
        result = ()
        score.astype(np.float)
        box = box.astype(np.int)
        if score < 0.4:
            continue

        cropped = img[box[1]:box[3], box[0]:box[2]]
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

def streamOnCammera():
    while True:
        isSuccess, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            t0 = time.time()
            res_obj = preprocess_face(frame, target_size=(112, 112))
            for img, box, score in res_obj:
                embedding = arcface_model.predict(img)
                embedding = embedding[0]
                score, source = search_face(embedding)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=1)
                cv2.putText(frame, source + str(score), (box[0], box[1] - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (255, 255, 255), 2)
        except:
            pass
        cv2.imshow("", frame[:, :, ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Terminate by user")
            break
        t1 = time.time()
        print("frame")
        print(f'took {round(t1 - t0, 3)} to process')
        # out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    # ---------- call class retina face detector --------------------------
    print("load retina face done!!")
    face_detector = RetinaFace(gpu_id=0)
    print("load ArcFace Models done!!")
    arcface_model = ArcFace.loadModel()
    es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
    streamOnCammera()

    # rent_video = input("Do You have Your Face in Data base yet ? yes : no \n")
    # if (isSuccess and rent_video == "yes"):

    # else:
    # print("Plase ADD you face to data base first ")
