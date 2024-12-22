import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# dics  del entrenamiento y test
data_dir = r"C:\\Users\\angel\\Desktop\\CUCEI\\INCO 9\\Modular\\codigo\\modular_project\\train_10"
test_dir = r"C:\\Users\\angel\\Desktop\\CUCEI\\INCO 9\\Modular\\codigo\\modular_project\\test10\\pruebas"

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)


train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# modelo con 4 convu
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# compilo  el modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback con pacienciade 10 epocas
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Evaluao
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = validation_generator.classes

# MX confusión
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.show()

# Reporte clssiifcacion
print("Reporte de Clasificación:\n")
print(classification_report(y_true, y_pred_classes, target_names=validation_generator.class_indices.keys()))

#  pérdida y precisión
plt.figure(figsize=(12, 5))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precision del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Precision')
plt.legend()

# loose
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.show()

# dics de clases
class_indices = train_generator.class_indices
classes = {v: k for k, v in class_indices.items()} 

def predict_image_with_probabilities(model, image_path):
    #carga la imagen
    image = load_img(image_path, target_size=(224, 224))
    plt.imshow(image)
    plt.title("Imagen de prueba")
    plt.axis("off")
    plt.show()

    # procesaar la imagen
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Añadir dimensión para lotes

    #hacer predicción
    predictions = model.predict(image_array)[0]  # Salida de probabilidades
    for i, prob in enumerate(predictions):
        print(f"{classes[i]}: {prob * 100:.2f}%")

# Probar del directorio de pruebas
for test_image in os.listdir(test_dir):
    test_image_path = os.path.join(test_dir, test_image)
    print(f"\nProcesando: {test_image}")
    predict_image_with_probabilities(model, test_image_path)

#Guardar el modelo entrenado
model_path = r"C:\\Users\\angel\\Desktop\\CUCEI\\INCO 9\\Modular\\codigo\\modular_project\\modelo_Person_11.h5"
model.save(model_path)
print(f"Modelo guardado como '{model_path}'.")
