from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

import keras
from tensorflow.keras import Sequential
print(tf.__version__)



app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# inference_layer = keras.saving.TFSMLayer('path/to/saved_model_directory', call_endpoint='serving_default')
#
# # Wrap the layer in a Sequential model
# MODEL = Sequential([inference_layer])
MODEL = tf.keras.models.load_model("../model.h5", compile=False)
# MODEL = tf.keras.models.load_model('path/to/saved_model_directory')
CLASS_NAMES =["Pepper bell Bacterial spot","Pepper bell healthy"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    # image = image.resize((224, 224))  # Example: (224, 224) for models like ResNet, VGG, etc.
    # # Convert image to array and normalize if necessary
    # image_array = np.array(image) / 255.0  # Normalize if model expects it
    # return image_array
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)