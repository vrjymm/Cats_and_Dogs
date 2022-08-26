import gradio as gr
import numpy as np
import tensorflow as tf
import PIL
import PIL.Image
import pickle


def greet_user(name):
    return f"Hello {name}, Welcome to Gradio!"

def return_multiple(name,number):
    return f"Hello {name}, The multiple of {number} is {2*number}."

def image_taker(image_path):
    
    input_image = tf.keras.utils.load_img(image_path, grayscale=False, color_mode='rgb', target_size=(256,256), interpolation='nearest', keep_aspect_ratio=False)

    input_image = tf.keras.utils.img_to_array(input_image)
    input = np.array([input_image]) 

    model = tf.keras.models.load_model('test.h5')
    predictions = model.predict(input)
    with open('label_map.pickle', 'rb') as handle:
        label_map = pickle.load(handle)

    return f"{label_map[predictions.argmax()]}"

app = gr.Interface(fn = image_taker, inputs=[gr.Image(type="filepath")], outputs="text")

app.launch()

