import cv2
import numpy as np
from tensorflow.keras.models import load_model


def preprocess_image(image_path, image_size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)

    if img is None:
        print(f"Error: The image at {image_path} could not be loaded.")
        return None

    img = cv2.resize(img, image_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    return img


model = load_model("./dist/shape-sorter.keras")
def predict_shape(image_path):
    img = preprocess_image(image_path)

    if img is None:
        return

    predictions = np.array(model.predict(img))

    print(f"Circle: {predictions[0][0] * 100}%")
    print(f"Square: {predictions[0][1] * 100}%")
    print(f"Triangle: {predictions[0][2] * 100}%")
    print(f"Pentagon: {predictions[0][3] * 100}%")
    print(f"Hexagon: {predictions[0][4] * 100}%")
    print(f"Star (Type 1): {predictions[0][5] * 100}%")
    print(f"Star (Type 2): {predictions[0][6] * 100}%")


if __name__ == "__main__":
    while True:
        predict_shape("./testing/test.png")
        print("Retry? (y/n)")
        retry = input()
        if retry != "y":
            exit()
