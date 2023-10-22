#Train.py
import tensorflow as tf
from Helper import Visualize,Data,Model


Percent=100
IMG_SIZE=(224,224)

Expt_Name="Image_"+str(Percent)
#Initalize path
raw_data_path="Data\Raw_image\\"
data_path="Data\\"+Expt_Name+"%\\"
train_path=data_path+"train\\"
val_path=data_path+"val\\"
test_path=data_path+"test\\"
label_path="JSON\\"+Expt_Name+".json"
checkpoint_path="Checkpoint\\"

labels,no_label=Data.get_labels(label_path=label_path).values()

train_data=tf.keras.preprocessing.image_dataset_from_directory(train_path,
                                                               label_mode="categorical",
                                                               image_size=IMG_SIZE,
                                                               shuffle=False)

val_data=tf.keras.preprocessing.image_dataset_from_directory(val_path,
                                                               label_mode="categorical",
                                                               image_size=IMG_SIZE,
                                                               shuffle=False)

test_data=tf.keras.preprocessing.image_dataset_from_directory(test_path,
                                                               label_mode="categorical",
                                                               image_size=IMG_SIZE,
                                                               shuffle=False)

#Create checkpoint
checkpoint_path=checkpoint_path+Expt_Name+"%"
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                       save_weights_only=True,
                                                       monitor="val_accuracy",
                                                       save_best_only=True)

#Import required models
data_augmentation=tf.keras.models.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomHeight(0.2),
    tf.keras.layers.RandomWidth(0.2)],
    name="Data_Augmentation")

base_model=tf.keras.applications.efficientnet.EfficientNetB0(include_top=False)
base_model.trainable=False

inputs=tf.keras.layers.Input(shape=(224,224,3), name="Input_Layer")

x=data_augmentation(inputs)

x=base_model(x,training=False)

x=tf.keras.layers.GlobalAveragePooling2D(name="Global_Average_Pooling")(x)

outputs=tf.keras.layers.Dense(no_label)(x)

model=tf.keras.Model(inputs,outputs)

model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

history=model.fit(train_data,
                  epochs=5,
                  validation_data=val_data,
                  validation_steps=len(val_data),
                  callbacks=[checkpoint_callback])

results=model.evaluate(test_data)

Visualize.plot_loss_curves(history)


