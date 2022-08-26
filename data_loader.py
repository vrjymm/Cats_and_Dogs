import numpy as np
import tensorflow as tf
import pickle

# create a data generator
datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)

# load and iterate training dataset
train_it = datagen.flow_from_directory('data', class_mode='categorical', batch_size=64, subset = "training")

#generate index to class label mapping
label_map = {v:k for k,v in train_it.class_indices.items()}

with open('label_map.pickle', 'wb') as handle:
    pickle.dump(label_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load and iterate validation dataset
val_it = datagen.flow_from_directory('data', class_mode='categorical', batch_size=64, subset = "validation")

# define model
input_t = tf.keras.Input(shape=(256,256,3))
pretrained_model = tf.keras.applications.ResNet50(include_top = False, weights="imagenet", input_tensor=input_t)
model = tf.keras.models.Sequential()
model.add(pretrained_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(2, activation='softmax'))


# fit model
model.compile(loss = 'categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=2e-5), metrics=['accuracy'] )
model.fit(train_it,epochs=2,steps_per_epoch=16, validation_data=val_it, validation_steps=8)

model.summary()

model.save('test_3.h5')
