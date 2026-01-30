import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
train_dir = "dataset/train"
val_dir = "dataset/val"
model_path = "model/currency_model.h5"

IMG_SIZE = 224
BATCH_SIZE = 16

# Data generators
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

model = load_model(model_path)

# Short retrain ONLY for metrics
history = model.fit(
    train_data,
    epochs=5,
    validation_data=val_data
)

# -------- Accuracy Graph --------
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.savefig('static/images/accuracy_graph.png')
plt.close()

# -------- Confusion Matrix --------
y_true = val_data.classes
y_pred = (model.predict(val_data) > 0.5).astype(int).reshape(-1)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
disp.plot()
plt.title("Confusion Matrix")
plt.savefig('static/images/confusion_matrix.png')
plt.close()

print("Graphs generated successfully")
