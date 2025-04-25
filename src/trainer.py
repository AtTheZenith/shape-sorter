import cv2
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_data(image_folder, image_size=(128, 128)):
    progress = 0
    progress_hun = 0
    images = []
    labels = []
    shape_labels = {"circle": 0, "square": 1, "triangle": 2}

    for shape in shape_labels.keys():
        shape_folder = os.path.join(image_folder, shape)
        for filename in os.listdir(shape_folder):
            progress += 1
            if filename.endswith(".png"):
                img = cv2.imread(
                    os.path.join(shape_folder, filename), cv2.IMREAD_COLOR_RGB
                )
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(shape_labels[shape])
            if progress - (progress % 100) > progress_hun:
                progress_hun = progress
                print(f"Progress: {progress_hun} images processed.")

    images = np.array(images)
    labels = np.array(labels)

    images = images.astype("float32") / 255.0

    images = np.expand_dims(images, axis=-1)

    labels = to_categorical(labels, num_classes=3)

    return images, labels


train_images, train_labels = load_data("training")
test_images, test_labels = load_data("testing")

x_train, x_val, y_train, y_val = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation="relu"),
    Dense(3, activation="softmax"),
])

model.compile(
    optimizer=Adam(learning_rate=0.0009),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop = EarlyStopping(
    monitor="val_accuracy", patience=2, restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "shape_recognition_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)

model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=12,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

model.save("./dist/shape-sorter.keras")
