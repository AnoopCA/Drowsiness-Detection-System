from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model
tf_model = Sequential()
tf_model.add(Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)))
tf_model.add(BatchNormalization())
tf_model.add(Conv2D(64, (3,3), activation='relu'))
tf_model.add(MaxPooling2D())
tf_model.add(Conv2D(128, (3,3), activation='relu'))
tf_model.add(Conv2D(128, (3,3), activation='relu'))
tf_model.add(MaxPooling2D())
tf_model.add(Conv2D(256, (3,3), activation='relu'))
tf_model.add(MaxPooling2D())
tf_model.add(Conv2D(256, (3,3), activation='relu'))
tf_model.add(MaxPooling2D())
tf_model.add(Flatten())
tf_model.add(Dense(256, activation='relu'))
tf_model.add(Dropout(0.3))
tf_model.add(Dense(1, activation='sigmoid'))
tf_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Preprocess and setup the data
train = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test = ImageDataGenerator(rescale=1./255)
train_path = r"D:\ML_Projects\Drowsiness-Detection-System\Data\Driver Drowsiness Dataset\train"
train_img = train.flow_from_directory(train_path, target_size=(256,256), batch_size=32, class_mode='binary')
test_path = r"D:\ML_Projects\Drowsiness-Detection-System\Data\Driver Drowsiness Dataset\test"
test_img = test.flow_from_directory(test_path, target_size=(256,256), batch_size=32, class_mode='binary')

# Train and test the model
mask_model = tf_model.fit(train_img, epochs=4, validation_data=test_img)

# Save the model
tf_model.save(r"D:\ML_Projects\Drowsiness-Detection-System\Models\drowse_model_tf_2.h5", mask_model)


