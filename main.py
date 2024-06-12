from fastapi import FastAPI, UploadFile, File , Response , status ,  HTTPException
from uvicorn import run
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow.keras as K
import tensorflow as tf
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import keras
import os
import cv2 as cv
import cloudinary
from cloudinary.uploader import upload

load_dotenv()

cloudinary.config(
    cloud_name=os.environ.get('CLOUD_NAME'),
    api_key=os.environ.get('API_KEY'),
    api_secret=os.environ.get('API_SECRET')
)

app = FastAPI()

model_dir = "model/leaf-disease.h5"
BETA_MODEL = load_model(model_dir)

origins = [
    "http://localhost:3000",
    "http://localhost"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def index() :
    return {"message":"Pinging"}

class_names = ['Apple scab','Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
'Cherry healthy','Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight','Corn healthy', 
'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy','Peach Bacterial spot','Peach healthy', 'Pepper bell Bacterial spot', 
'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Strawberry Leaf scorch', 'Strawberry healthy',
'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

def is_leaf_present(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 1000:
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) > 4:
                return True
    return False

def upload_to_cloudinary(image_bytes):
    try:
        response = upload(image_bytes, folder='leaf_uploads')
        return response['url']
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail="Failed to upload image to Cloudinary: " + str(e))

def mark_disease(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask = cv.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 1000:
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) > 4:
                cv.drawContours(image, [contour], -1, (0, 0, 255), 2) 
    return image

def generate_heatmap(image, probabilities):
    heatmap = cv.resize(probabilities, (image.shape[1], image.shape[0]))
    heatmap = cv.applyColorMap((heatmap * 255).astype(np.uint8), cv.COLORMAP_JET)
    heatmap = cv.addWeighted(heatmap, 0.5, image, 0.5, 0)
    return heatmap

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        bytes = await file.read()
        img = cv.imdecode(np.frombuffer(bytes, dtype=np.uint8), cv.IMREAD_COLOR)
        is_leaf = is_leaf_present(img)
        if is_leaf:
            normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)
            predictions = BETA_MODEL.predict(normalized_image)
            confidence = np.max(predictions[0]) * 100
            class_index = np.argmax(predictions)
            class_name = class_names[class_index]

            # -- marker image

            marked_image = mark_disease(img)
            marked_image_bytes = cv.imencode('.jpg', marked_image)[1].tobytes()
            cloudinary_marked_url = upload_to_cloudinary(marked_image_bytes)

            # -- heatmap image

            probability_map = predictions[0]
            heatmap_overlay = generate_heatmap(marked_image, probability_map)
            heatmap_bytes = cv.imencode('.jpg', heatmap_overlay)[1].tobytes()
            # cloudinary_heatmap_url = upload_to_cloudinary(heatmap_bytes)


            # imageUrl = upload_to_cloudinary(bytes)
           
            # -- res

             # return {'class_name': class_name, 'confidence': confidence, 'imageUrl' : imageUrl , 'markedUrl': cloudinary_marked_url, 'heatmapUrl': cloudinary_heatmap_url}    
            
            #! -- Do not push to PROD
            return {'class_name' : class_name , 'confidence' : confidence , 'imageUrl' : 'http://res.cloudinary.com/dh1isn914/image/upload/v1711437754/leaf_uploads/iajmgoz1rdqs3dq6xvbd.jpg','markedUrl':"http://res.cloudinary.com/dh1isn914/image/upload/v1711437748/leaf_uploads/hrusytnophva5lva2fji.jpg",'heatmapUrl':"http://res.cloudinary.com/dh1isn914/image/upload/v1711437754/leaf_uploads/k1fne9cqnzg2c2okubtl.jpg"}
            # return {'class_name' : class_name , 'confidence' : confidence , 'imageUrl' : 'http://res.cloudinary.com/dh1isn914/image/upload/v1711434036/leaf_uploads/qlpdpl6v4a67ee0accsb.jpg','markedUrl':"http://res.cloudinary.com/dh1isn914/image/upload/v1711434035/leaf_uploads/dtq5v1jxjasjzvvrxeh1.jpg",'heatmapUrl':"http://res.cloudinary.com/dh1isn914/image/upload/v1711434036/leaf_uploads/lmdxtin7sgbdzn8uivvg.jpg"}
        else:
            raise HTTPException(status_code=500, detail='No leaf detected')
    except Exception as e:
            raise HTTPException(status_code=500, detail="Image Cannot Be Processed: "+str(3))
    

    
if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	run(app, host="0.0.0.0", port=port)

