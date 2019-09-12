import tensorflow as tf
import wandb

# logging code
run = wandb.init()
config = run.config

# load data
# x is input, y is output (labels)
# X_train = (6000, 28, 28) 6000 28x28 arrays
# y_train 60000 images
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# print(X_test.shape)
# exit()
img_width = X_train.shape[1]
img_height = X_train.shape[2]

# normalize data
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# one hot encode outputs
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
labels = [str(i) for i in range(10)]

# reshape input data
X_train = X_train.reshape(
    X_train.shape[0], img_width, img_height, 1)
X_test = X_test.reshape(
    X_test.shape[0], img_width, img_height, 1)

num_classes = y_train.shape[1]


# create model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3, 3), input_shape=(img_width, img_height, 1)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(16, (3, 3)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
# you optimize the loss function
# mse = optimize the difference between the output and activation function
# model.compile(loss='mse', optimizer='adam',
#               metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# Fit the model

# if high accuracy on training data and low on validation data, we have overfitting
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
          callbacks=[wandb.keras.WandbCallback(data_type="image", labels=labels, save_model=False)])

