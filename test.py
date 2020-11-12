import os
import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np


interpreter = tflite.Interpreter(model_path='model.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cats = []
dogs = []

for filename in os.listdir('Images'):   
    img = Image.open('Images/'+ filename)
    img = img.resize((200,200))
    np_arr = np.asarray(img)

    input_data = np.array(np_arr, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    if output_data[0] > 0.5:
        dogs.append(filename+' ' + str(output_data[0]))
    else:
        cats.append(filename+' ' + str(output_data[0]))

print('Cats: ' + str(cats))
print('Dogs: ' + str(dogs))