import tensorflow as tf
import wandb

wandb.init()

config = wandb.config
config.batch_size = 128
config.epochs = 10
config.learn_rate = 0.001

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(class_names)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalization
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# print(y_train.shape)
# exit()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(10, (3, 3), padding='same',
                                 input_shape=X_train.shape[1:],
                                 activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(20, (3, 3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten(input_shape=X_train.shape[1:]))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

# for classification, categorical cross entropy is a better loss equation
# it makes the effect output its true belief regarding the probabilities
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(config.learn_rate),
              metrics=['accuracy'])
# log the number of total parameters
config.total_params = model.count_params()
print("Total params: ", config.total_params)

# Sources of nondeterminism conditions:

model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test),
          callbacks=[wandb.keras.WandbCallback(data_type="image", labels=class_names, save_model=False)])

