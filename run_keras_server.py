from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.models import load_model
import numpy as np
from PIL import Image
import flask
from io import BytesIO
import base64


app = flask.Flask(__name__)
model = None

# def load_model():
#     global model
#     model = load_model('modelos/model_nivel1_VGG.h50')
np
def prepare_image(image, target):
    print(image, target)
    image = image.resize(target)
    print('a')
    image = img_to_array(image)
    print('a1')
    image = np.expand_dims(image, axis=0)
    print('a2')
    image = imagenet_utils.preprocess_input(image)
    print('a3')
    return image

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    imageb64 = flask.request.json['image']

    im = Image.open(BytesIO(base64.b64decode(imageb64)))
    im = im.convert(mode='RGB')
    print(im)
    image64x64 = prepare_image(im, target=(64, 64))
    print(image64x64)
    global model
    model = load_model('modelos/model_nivel1_VGG.h5')

    # print(model.predict(image64x64))
    # print(im)
    # if flask.request.method == "POST":
        # image = flask.request.form['image']
        # print(image)
        # if flask.request.files.get("image"):
        #     image = flask.request.files["image"].read()
        #     data["success"] = True
        #     data["image"] = image
    result = model.predict(image64x64, batch_size=1)
    print(result)
    return flask.jsonify(data)

if __name__ == "__main__":
    app.run()