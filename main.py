from fastapi import FastAPI
from fastapi import Body
from pydantic import BaseModel
import face

app = FastAPI()


class PicItem(BaseModel):
    pic: str


# 获取人脸特征向量
@app.post("/get_face_embedding")
async def get_face_embedding(pic_item: PicItem):
    embeddings = face.get_face_embedding_normalize_base64(pic_item.pic)
    results = []
    for embedding in embeddings:
        results.append(embedding.tolist())
    return results
