from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import cv2
import base64
from src.face_detection import detect_and_predict_emotion

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------- Predict from uploaded image ----------
@app.post("/predict_image/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results, processed_frame = detect_and_predict_emotion(frame)

    # Convert processed image to Base64 to send it back to frontend
    _, buffer = cv2.imencode('.jpg', processed_frame)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse({
        "results": results,
        "image": f"data:image/jpeg;base64,{img_str}"
    })


# ---------- Predict from live camera ----------
@app.post("/predict_camera/")
async def predict_camera(request: Request):
    data = await request.json()
    image_data = data["image"].split(",")[1]
    image_bytes = base64.b64decode(image_data)
    np_img = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results, processed_frame = detect_and_predict_emotion(frame)

    _, buffer = cv2.imencode('.jpg', processed_frame)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse({
        "results": results,
        "image": f"data:image/jpeg;base64,{img_str}"
    })
