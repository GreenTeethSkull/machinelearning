#Parte I - Construir el modelo de CNN

#importar las librerías y paquetas
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Inicializar la CNN
classifier = Sequential()

#Paso 1 - Convolución
classifier.add(Convolution2D(filters = 32,kernel_size = 3,input_shape = (64,64,3), activation = "relu"))

#Paso 2 - Mix Pooling
classifier.add(MaxPooling2D(pool_size = 2))

#Mejora del modelo
classifier.add(Convolution2D(filters = 32,kernel_size = 3, activation = "relu"))
classifier.add(MaxPooling2D(pool_size = 2))

#Paso 3 - Flattening
classifier.add(Flatten())

#Paso 4 - Full Connection
classifier.add(Dense(units = 128, activation = "relu" ))
classifier.add(Dense(units = 1, activation = "sigmoid" ))

#Compilar la CNN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#Parte 2 - Ajustar la CNN a las imágenes para entrenar
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit(
        train_generator,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=validation_generator,
        validation_steps=2000)