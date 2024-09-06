import base64
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# 加载模型
app = FaceAnalysis(providers='CPUExecutionProvider')
app.prepare(ctx_id=-1)
face_dic = {}


def feature_compare(feature1, feature2):
    # diff = np.subtract(feature1, feature2)
    # dist = np.sum(np.square(diff), 1)
    similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    similarity_adjusted = (similarity + 1) / 2  # 将范围从 [-1, 1] 转换到 [0, 1]
    similarity_percentage = similarity_adjusted * 100  # 转换到百分比
    return similarity_percentage


def get_face_embedding_normalize_base64(base64_str):
    decoded_data = base64.b64decode(base64_str)
    nparr = np.fromstring(decoded_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces = app.get(img)
    faces_embedding = []
    for face in faces:
        faces_embedding.append(face.normed_embedding)
    return faces_embedding

#识别人脸
def recognize(base64_str):
    decoded_data = base64.b64decode(base64_str)
    nparr = np.fromstring(decoded_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces = app.get(img)
    face_embedding = faces[0].normed_embedding
    # face_embedding = np.array(face_embedding).reshape((1, -1))
    # face_embedding = preprocessing.normalize(face_embedding)
    reco_dict = {}
    for face_name in face_dic:
        reco_dict[face_name] = feature_compare(face_dic[face_name], face_embedding)
    return reco_dict

