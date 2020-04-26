import numpy as np
from PIL import Image
import flask
from io import BytesIO
import base64

from preprocessing import ImageToArrayPreprocessor
from preprocessing import AspectAwarePreprocessor
from preprocessing.meanpreprocessor import MeanPreprocessor
from datasets import SimpleDatasetLoader

import tensorflow as tf

from keras import backend as K
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.models import load_model

app = flask.Flask(__name__)
model = None

def load_model_from_path(path):
    graph = tf.get_default_graph()
    model = load_model(path)
    return graph, model

def load_all_models():
    global g_model_objs
    g_model_objs = dict()

    g_model_objs = {
        'model_nivel1': load_model_from_path('models/model_nivel1_VGG.h5'),
        'model_nivel2_suco': load_model_from_path('models/model_nivel2_LeNetSuco.h5'),
        'model_nivel2_refrigerante': load_model_from_path('models/model_nivel2_LeNetRefrigerante.h5'),
        'model_nivel3_cocacola': load_model_from_path('models/model_nivel3_CocaCola.h5'),
        'model_nivel3_guarana': load_model_from_path('models/model_nivel3_Guarana.h5'),
        'model_nivel3_fanta': load_model_from_path('models/model_nivel3_Fanta.h5'),
        'model_nivel3_pepsi': load_model_from_path('models/model_nivel3_Pepsi.h5'),
        'model_nivel3_dobem': load_model_from_path('models/model_nivel3_DoBem.h5'),
        'model_nivel3_delvalle': load_model_from_path('models/model_nivel3_DelValle.h5'),
        'model_nivel3_maguary': load_model_from_path('models/model_nivel3_Maguary.h5'),
        'model_nivel3_carrefour': load_model_from_path('models/model_nivel3_Carrefour.h5')
    }

def processing_image(image, target):
    aap = AspectAwarePreprocessor(target[0],target[1])
    iap = ImageToArrayPreprocessor()
    mpp = MeanPreprocessor(123.68, 116.779, 103.939)

    sdl_nivel1 = SimpleDatasetLoader(preprocessors=[aap, iap, mpp])
    sdl_nivelx = SimpleDatasetLoader(preprocessors=[aap, iap])

    X_test_nivel1 = sdl_nivel1.preprocess(image)
    X_test_nivelx = sdl_nivelx.preprocess(image)

    X_test_nivel1 = X_test_nivel1.astype("float") / 255.0
    X_test_nivelx = X_test_nivelx.astype("float") / 255.0

    return X_test_nivel1, X_test_nivelx


# Asignação de nomes aos rótulos para cada classificador
class_hierarchy = {
    "produto": {0: "refrigerante", 1: "suco"},
    "suco": {0: "carrefour", 1: "delValle", 2: "doBem", 3: "maguary" },
    "refrigerante": {0: "coca-cola", 1: "fanta", 2: "guarana", 3: "pepsi"},
    "doBem": {0: "limonada-doBem", 1: "manga-doBem", 2: "pessego-doBem", 3: "uva-doBem"},
    "delValle": {0: "abacaxi-delValle", 1: "caju-delValle", 2: "laranja-delValle", 3: "maracuja-delValle", 4: "uva-delValle"},
    "maguary": {0: "goiaba-maguary", 1: "manga-maguary", 2: "maracuja-maguary", 3: "morango-maguary", 4: "uva-maguary"},
    "carrefour": {0: "abacaxi-carrefour", 1: "laranja-carrefour", 2: "pessego-maca-carrefour", 3: "uva-carrefour"},
    "coca-cola": {0: "garrafa-original-cc", 1: "garrafa-semacucar-cc", 2: "lata-original-cc", 3: "lata-semacucar-cc"},
    "fanta": {0: "garrafa-laranja-fanta", 1: "lata-guarana-fanta", 2: "lata-laranja-fanta"},
    "guarana": {0: "garrafa-original-guarana", 1: "lata-antarctica-guarana", 2: "lata-original-guarana"},
    "pepsi": {0: "garrafa-original-pepsi", 1: "garrafa-twist-pepsi", 2: "lata-original-pepsi", 3: "lata-semacucar-pepsi"},
}

load_all_models()

def prediction_by_model(image, graph, model):
    X = image
    try:
        with graph.as_default():
            print("fnjsrrgorghiarowsarjghaowefjoapefkaejaf")
            predictions = model.predict(X)
            print("Prediction: ", predictions)
            return predictions
    except Exception as err:
        raise(err)
    return None

def run_predictions(image1, imagex):
    predictions = []
    # nivel 1
    graph, model_nivel1 = g_model_objs['model_nivel1']
    try:
        produto = prediction_by_model(image1, graph, model_nivel1)
        y_predict_index = np.argmax(produto)
        predictions.append(class_hierarchy["produto"][y_predict_index])
    except Exception as error:
        print("Error in predict")
    #nivel 2
    if y_predict_index == 0: # refrigerante
        graph, model_nivel2 = g_model_objs['model_nivel2_refrigerante']
        # 0: coca-cola
        # 1: fanta
        # 2: guarana
        # 3: pepsi
        refrigerante = prediction_by_model(imagex, graph, model_nivel2)
        y2_predict_index = np.argmax(refrigerante)
        predictions.append(class_hierarchy["refrigerante"][y2_predict_index])
    elif y_predict_index == 1: # suco
        graph, model_nivel2 = g_model_objs['model_nivel2_suco']
        # 0: carrefour
        # 1: delvalle
        # 2: dobem
        # 3: maguary
        suco = prediction_by_model(imagex, graph, model_nivel2)
        y2_predict_index = np.argmax(suco)
        predictions.append(class_hierarchy["suco"][y2_predict_index])

    #nivel 3
    if y_predict_index == 0: # refrigerante
        if y2_predict_index == 0: # marcas de refrigerante
            graph, model_nivel3 = g_model_objs['model_nivel3_cocacola']
            # 0: garrafa-original600ml
            # 1: garrafa-semacucar600ml
            # 2: lata-original310ml
            # 3: lata-semacucar220ml
            coca = prediction_by_model(imagex, graph, model_nivel3)
            y3_predict_index = np.argmax(coca)
            predictions.append(class_hierarchy["coca-cola"][y3_predict_index])
        elif y2_predict_index == 1:
            graph, model_nivel3 = g_model_objs['model_nivel3_fanta']
            # 0: garrafa-laranja200ml
            # 1: lata-guarana350ml
            # 2: lata-laranja350ml
            fanta = prediction_by_model(imagex, graph, model_nivel3)
            y3_predict_index = np.argmax(fanta)
            predictions.append(class_hierarchy["fanta"][y3_predict_index])
        elif y2_predict_index == 2:
            graph, model_nivel3 = g_model_objs['model_nivel3_guarana']
            # 0: garrafa-laranja600ml
            # 1: lata-antarctica350ml
            # 2: lata-original350ml
            guarana = prediction_by_model(imagex, graph, model_nivel3)
            y3_predict_index = np.argmax(guarana)
            predictions.append(class_hierarchy["guarana"][y3_predict_index])
        elif y2_predict_index == 3:
            graph, model_nivel3 = g_model_objs['model_nivel3_pepsi']
            # 0: garrafa-original
            # 1: garrafa-twist600ml
            # 2: lata-original350ml
            # 3: lata-semacucar350ml
            pepsi = prediction_by_model(imagex, graph, model_nivel3)
            y3_predict_index = np.argmax(pepsi)
            predictions.append(class_hierarchy["pepsi"][y3_predict_index])
    elif y_predict_index == 1: # suco
        if y2_predict_index == 0: # marcas de suco
            graph, model_nivel3 = g_model_objs['model_nivel3_carrefour']
            # 0: abacaxi
            # 1: laranja
            # 2: pessego-maça
            # 3: uva
            carrefour = prediction_by_model(imagex, graph, model_nivel3)
            y3_predict_index = np.argmax(carrefour)
            predictions.append(class_hierarchy["carrefour"][y3_predict_index])
        elif y2_predict_index == 1:
            raph, model_nivel3 = g_model_objs['model_nivel3_delvalle']
            # 0: abacaxi
            # 1: caju
            # 2: laranja
            # 3: maracuja
            # 4: uva
            delvalle = prediction_by_model(imagex, graph, model_nivel3)
            y3_predict_index = np.argmax(delvalle)
            predictions.append(class_hierarchy["delValle"][y3_predict_index])
        elif y2_predict_index == 2:
            graph, model_nivel3 = g_model_objs['model_nivel3_dobem']
            # 0: limonada
            # 1: manga
            # 2: pessego
            # 3: uva
            dobem = prediction_by_model(imagex, graph, model_nivel3)
            y3_predict_index = np.argmax(dobem)
            predictions.append(class_hierarchy["doBem"][y3_predict_index])
        elif y2_predict_index == 3:
            graph, model_nivel3 = g_model_objs['model_nivel3_maguary']
            # 0: goiaba
            # 1: manga
            # 2: maracuja
            # 3: morango
            # 4: uva
            maguary = prediction_by_model(imagex, graph, model_nivel3)
            y3_predict_index = np.argmax(maguary)
            predictions.append(class_hierarchy["maguary"][y3_predict_index])
    return predictions

@app.route("/predict", methods=["POST"])
def predict():
    
    data = {"success": False}
    imageb64 = flask.request.json['image']

    image = Image.open(BytesIO(base64.b64decode(imageb64)))
    image = image.convert(mode='RGB')
    image = np.array(image)
    print(np.array(image).shape)
    image1, imagex = processing_image(image, target=(64,64))
    print(np.array(image1).shape)
    print(np.array(imagex).shape)
    # get detected product
    predictions = run_predictions(image1, imagex)

    

    # print(model.predict(image64x64))
    # print(im)
    # if flask.request.method == "POST":
        # image = flask.request.form['image']
        # print(image)
        # if flask.request.files.get("image"):
        #     image = flask.request.files["image"].read()
        #     data["success"] = True
        #     data["image"] = image
    
    data["success"] = True
    data["Prediction Nivel 1"] = predictions[0]
    data["Prediction Nivel 2"] = predictions[1]
    data["Prediction Nivel 3"] = predictions[2]
    K.clear_session()
    print(data)
    return flask.jsonify(data)

if __name__ == "__main__":
    app.run()