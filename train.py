import pandas as pd
from PIL import Image
import numpy as np
import os
import sklearn
import tensorflow as tf
#just some playing around with one image

# image = Image.open("chest.jpg")
# arr = np.array(image)
# print(type(image)) #type of image ->  here its a PIL type jpeg image
# print(type(arr)) #numpy array
# print(arr.shape)
#---------------------------------------------------------------------------------------------------------------------------

#folder = r"C:\Users\judad\Downloads\pneumonia\chest_xray" #path of where my data is
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
def load_dataset(folder): #here folder will be leading to train,val or test
    X = []
    y = []
    classes = {
        "NORMAL" : 0,
        "PNEUMONIA" : 1
    }
    for class_name,label in classes.items(): #.items allows to view both key and value pairs at the same time
        class_path = os.path.join(folder,class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path,image_name)

            try:
                img = Image.open(image_path)
                img = img.convert("RGB")
                img = img.resize((224,224))

                arr = np.array(img)
                arr = preprocess_input(arr)
                X.append(arr)
                y.append(label)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
    X = np.array(X)
    y = np.array(y)
    return X,y
from sklearn.model_selection import train_test_split

train_path = r"C:\Users\judad\Downloads\pneumonia\chest_xray\train"
val_path = r"C:\Users\judad\Downloads\pneumonia\chest_xray\val"
test_path = r"C:\Users\judad\Downloads\pneumonia\chest_xray\test"

X_train,y_train = load_dataset(train_path)
X_val,y_val = load_dataset(val_path)
X_test,y_test = load_dataset(test_path)

X_all = np.concatenate([X_train,X_val])
y_all = np.concatenate([y_train,y_val])


X_train,X_val,y_train,y_val = train_test_split(X_all,y_all,test_size = 0.2, random_state=42,stratify=y_all)

# X_train = np.expand_dims(X_train, axis=-1)
# X_val = np.expand_dims(X_val, axis=-1)
# X_test = np.expand_dims(X_test, axis=-1)

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_val, return_counts=True))
print(np.unique(y_test, return_counts=True))

#TRAINING PART---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,GlobalAveragePooling2D,Dropout
from tensorflow.keras import Input
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
# modification = ImageDataGenerator(
#     height_shift_range = 0.05,
#     width_shift_range = 0.05,
#     zoom_range = 0.05,
#     rotation_range = 5
# )

# train_generator = modification.flow(
#     X_train,y_train,batch_size = 32
# )
base_model = MobileNetV2(
    include_top = False,
    weights = "imagenet",
    input_shape = (224,224,3)
)

base_model.trainable = False
from tensorflow.keras.layers import BatchNormalization
model = Sequential([
    Input(shape = (224,224,3)),
    base_model,
    # Conv2D(filters = 32,kernel_size=(3,3),activation="relu"),
    # MaxPooling2D(pool_size = (2,2)),
    # Conv2D(filters = 64,kernel_size = (3,3),activation = "relu"),
    # MaxPooling2D(pool_size = (2,2)),
    # Conv2D(filters = 64,kernel_size = (3,3),activation = "relu"),
    # MaxPooling2D(pool_size = (2,2)),
    GlobalAveragePooling2D(),
    Dense(units = 128,activation = "relu"),
    # BatchNormalization(),
    Dropout(0.3),
    Dense(units = 1,activation = "sigmoid")

])
from tensorflow.keras.optimizers import Adam
model.summary()
model.compile(
    optimizer = "adam",
    metrics = ["accuracy"],
    loss = "binary_crossentropy"
)
from tensorflow.keras.callbacks import EarlyStopping


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

weights = compute_class_weight(
    classes = np.unique(y_train),
    y=y_train,
    class_weight="balanced"
)

class_weights = {
    0 : weights[0],
    1 : weights[1]
}

history = model.fit(
    X_train,y_train,batch_size = 32,
    validation_data = (X_val,y_val),
    epochs = 20,
    callbacks = [early_stop],
    class_weight = class_weights
)

test_loss, test_acc = model.evaluate(X_test, y_test)

print("Test Accuracy:", test_acc)
from sklearn.metrics import confusion_matrix, classification_report

y_pred_prob = model.predict(X_test)
for t in [0.5,0.55,0.6,0.65]:
    y_pred = (y_pred_prob > t).astype(int)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

model.save("penumonia-classifier.keras")
print("model saved successfully")
