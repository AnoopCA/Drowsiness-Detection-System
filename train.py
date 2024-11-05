from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

start_time = time.time()

# Define the model
tf_model = Sequential()
tf_model.add(Conv2D(32, (3,3), activation='relu', input_shape=(80, 80, 1)))
tf_model.add(BatchNormalization())
tf_model.add(Conv2D(64, (3,3), activation='relu'))
tf_model.add(MaxPooling2D())
tf_model.add(Conv2D(128, (3,3), activation='relu'))
tf_model.add(Conv2D(128, (3,3), activation='relu'))
tf_model.add(MaxPooling2D())
tf_model.add(Conv2D(256, (3,3), activation='relu'))
tf_model.add(Conv2D(256, (3,3), activation='relu'))
tf_model.add(MaxPooling2D())
tf_model.add(Flatten())
tf_model.add(Dense(10*10*256, activation='relu'))
tf_model.add(Dropout(0.3))
tf_model.add(Dense(784, activation='relu'))
tf_model.add(Dense(1, activation='sigmoid'))
tf_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Preprocess and setup the data
train = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test = ImageDataGenerator(rescale=1./255)
train_path = r"D:\ML_Projects\Drowsiness-Detection-System\Data\MRL_Eye\train"
train_img = train.flow_from_directory(train_path, target_size=(80,80), batch_size=512, class_mode='binary', color_mode='grayscale') #512-1:48
test_path = r"D:\ML_Projects\Drowsiness-Detection-System\Data\MRL_Eye\test"
test_img = test.flow_from_directory(test_path, target_size=(80,80), batch_size=512, class_mode='binary', color_mode='grayscale') #512

# Train and test the model
mask_model = tf_model.fit(train_img, epochs=2, validation_data=test_img)

# Save the model
tf_model.save(r"D:\ML_Projects\Drowsiness-Detection-System\Models\drowse_model_tf_4.h5", mask_model)

end_time = time.time()

execution_time = end_time - start_time
hours, remainder = divmod(execution_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Execution time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
