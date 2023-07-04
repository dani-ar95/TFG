import tensorflow as tf
import os
import cv2 as cv

# load images in a list
X_train = []

imagenes = os.listdir("/home/daniel/TFG/FireExtinguisher-2/train/images")

for i in range(100):
    try:
      img = imagenes[i]
        
      img = cv.imread(os.path.join("/home/daniel/TFG/FireExtinguisher-2/train/images/", img), cv.IMREAD_COLOR)
      img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
      img = cv.resize(img, (800, 800), cv.INTER_AREA)
      img = img / 255.0
      X_train.append(img)

    except Exception as e:
        pass

print(len(X_train))

model = tf.keras.models.load_model("runs/detect/train/weights/best_saved_model")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter = tf.lite.TFLiteConverter.from_saved_model("runs/detect/train/weights/best_saved_model")

def representative_dataset_gen():
    for data in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(100):
        yield [tf.dtypes.cast(data, tf.float32)]


converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_gen)


#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

tflite_model = converter.convert()

# Save the model.
with open('converted_model.tflite', 'wb') as f:
    f.write(tflite_model)



