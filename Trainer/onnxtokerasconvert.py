import onnx2keras
from onnxtokerasconvert import onnx_to_keras
import keras
import onnx

onnx_model = onnx.load('best.onnx')
k_model = onnx_to_keras(onnx_model, ['images'])

keras.models.save_model(k_model,'kerasModel.h5',overwrite=True,include_optimizer=True)