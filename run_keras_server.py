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

# Asignação de nomes aos rótulos para cada classificador
class_hierarchy = {
    "produto"     : { 0: "refrigerante", 
                      1: "suco" 
                    },
    "suco"        : { 0: "carrefour", 
                      1: "delValle", 
                      2: "doBem", 
                      3: "maguary" 
                    },
    "refrigerante": { 0: "coca-cola",
                      1: "fanta", 
                      2: "guarana", 
                      3: "pepsi"
                    },
    "doBem"       : { 0: "limonada-doBem", 
                      1: "manga-doBem", 
                      2: "pessego-doBem", 
                      3: "uva-doBem"
                    },
    "delValle"    : { 0: "abacaxi-delValle", 
                      1: "caju-delValle", 
                      2: "laranja-delValle", 
                      3: "maracuja-delValle", 
                      4: "uva-delValle"
                    },
    "maguary"     : { 0: "goiaba-maguary", 
                      1: "manga-maguary", 
                      2: "maracuja-maguary", 
                      3: "morango-maguary", 
                      4: "uva-maguary"
                    },
    "carrefour"   : { 0: "abacaxi-carrefour", 
                      1: "laranja-carrefour", 
                      2: "pessego-maca-carrefour", 
                      3: "uva-carrefour"
                    },
    "coca-cola"   : { 0: "garrafa-original-cc", 
                      1: "garrafa-semacucar-cc", 
                      2: "lata-original-cc", 
                      3: "lata-semacucar-cc"
                    },
    "fanta"       : { 0: "garrafa-laranja-fanta", 
                      1: "lata-guarana-fanta", 
                      2: "lata-laranja-fanta"
                    },
    "guarana"     : { 0: "garrafa-original-guarana", 
                      1: "lata-antarctica-guarana", 
                      2: "lata-original-guarana"
                    },
    "pepsi"       : { 0: "garrafa-original-pepsi", 
                      1: "garrafa-twist-pepsi", 
                      2: "lata-original-pepsi", 
                      3: "lata-semacucar-pepsi"
                    }
}


def load_model_from_path(path):
    graph = tf.get_default_graph()
    model = load_model(path)
    model._make_predict_function()
    return graph, model

def load_all_models():
    global g_model_objs
    g_model_objs = dict()

    g_model_objs = {
        # 'model_nivel1'             : load_model_from_path('models/model_nivel1_VGG.h5'),
        'model_nivel1'             : load_model_from_path('models/model_nivel1_LeNet.h5'),
        'model_nivel2_suco'        : load_model_from_path('models/model_nivel2_LeNetSuco.h5'),
        'model_nivel2_refrigerante': load_model_from_path('models/model_nivel2_LeNetRefrigerante.h5'),
        'model_nivel3_cocacola'    : load_model_from_path('models/model_nivel3_CocaCola.h5'),
        'model_nivel3_guarana'     : load_model_from_path('models/model_nivel3_Guarana.h5'),
        'model_nivel3_fanta'       : load_model_from_path('models/model_nivel3_Fanta.h5'),
        'model_nivel3_pepsi'       : load_model_from_path('models/model_nivel3_Pepsi.h5'),
        'model_nivel3_dobem'       : load_model_from_path('models/model_nivel3_DoBem.h5'),
        'model_nivel3_delvalle'    : load_model_from_path('models/model_nivel3_DelValle.h5'),
        'model_nivel3_maguary'     : load_model_from_path('models/model_nivel3_Maguary.h5'),
        'model_nivel3_carrefour'   : load_model_from_path('models/model_nivel3_Carrefour.h5')
    }



def prediction_by_model(image, graph, model):
    with graph.as_default():
        predictions = model.predict(image)
        return predictions

def nivel1_predict(image):
    graph, model = g_model_objs['model_nivel1']
    produto = prediction_by_model(image, graph, model)
    
    return np.argmax(produto)

def nivel2_predict(image, y1_predict_index):
    y2_predict_index = -1
    if y1_predict_index == 0: # refrigerante
        graph, model_nivel2 = g_model_objs['model_nivel2_refrigerante']
        # 0: coca-cola
        # 1: fanta
        # 2: guarana
        # 3: pepsi
        refrigerante = prediction_by_model(image, graph, model_nivel2)
        y2_predict_index = np.argmax(refrigerante)
        
        # predictions.append(class_hierarchy["refrigerante"][y2_predict_index])
    elif y1_predict_index == 1: # suco
        graph, model_nivel2 = g_model_objs['model_nivel2_suco']
        # 0: carrefour
        # 1: delvalle
        # 2: dobem
        # 3: maguary
        suco = prediction_by_model(image, graph, model_nivel2)
        y2_predict_index = np.argmax(suco)
        
        # predictions.append(class_hierarchy["suco"][y2_predict_index])
    
    return y2_predict_index

def nivel3_predict(image, y1_predict_index, y2_predict_index):
    y3_predict_index = -1
    if y1_predict_index == 0: # refrigerante
        if y2_predict_index == 0: # coca-cola
            graph, model_nivel3 = g_model_objs['model_nivel3_cocacola']
            # 0: garrafa-original600ml
            # 1: garrafa-semacucar600ml
            # 2: lata-original310ml
            # 3: lata-semacucar220ml
            coca = prediction_by_model(image, graph, model_nivel3)
            y3_predict_index = np.argmax(coca)
            
        elif y2_predict_index == 1:
            graph, model_nivel3 = g_model_objs['model_nivel3_fanta']
            # 0: garrafa-laranja200ml
            # 1: lata-guarana350ml
            # 2: lata-laranja350ml
            fanta = prediction_by_model(image, graph, model_nivel3)
            y3_predict_index = np.argmax(fanta)
            
        elif y2_predict_index == 2:
            graph, model_nivel3 = g_model_objs['model_nivel3_guarana']
            # 0: garrafa-laranja600ml
            # 1: lata-antarctica350ml
            # 2: lata-original350ml
            guarana = prediction_by_model(image, graph, model_nivel3)
            y3_predict_index = np.argmax(guarana)
            
        elif y2_predict_index == 3:
            graph, model_nivel3 = g_model_objs['model_nivel3_pepsi']
            # 0: garrafa-original
            # 1: garrafa-twist600ml
            # 2: lata-original350ml
            # 3: lata-semacucar350ml
            pepsi = prediction_by_model(image, graph, model_nivel3)
            y3_predict_index = np.argmax(pepsi)
            
    elif y1_predict_index == 1: # suco
        if y2_predict_index == 0: # marcas de suco
            graph, model_nivel3 = g_model_objs['model_nivel3_carrefour']
            # 0: abacaxi
            # 1: laranja
            # 2: pessego-maça
            # 3: uva
            carrefour = prediction_by_model(image, graph, model_nivel3)
            y3_predict_index = np.argmax(carrefour)
            
        elif y2_predict_index == 1:
            graph, model_nivel3 = g_model_objs['model_nivel3_delvalle']
            # 0: abacaxi
            # 1: caju
            # 2: laranja
            # 3: maracuja
            # 4: uva
            delvalle = prediction_by_model(image, graph, model_nivel3)
            y3_predict_index = np.argmax(delvalle)
            
        elif y2_predict_index == 2:
            graph, model_nivel3 = g_model_objs['model_nivel3_dobem']
            # 0: limonada
            # 1: manga
            # 2: pessego
            # 3: uva
            dobem = prediction_by_model(image, graph, model_nivel3)
            y3_predict_index = np.argmax(dobem)
            
        elif y2_predict_index == 3:
            graph, model_nivel3 = g_model_objs['model_nivel3_maguary']
            # 0: goiaba
            # 1: manga
            # 2: maracuja
            # 3: morango
            # 4: uva
            maguary = prediction_by_model(image, graph, model_nivel3)
            y3_predict_index = np.argmax(maguary)
    
    return y3_predict_index

def processing_image(image):
    aap = AspectAwarePreprocessor(64, 64)
    iap = ImageToArrayPreprocessor()

    sdl_nivel = SimpleDatasetLoader(preprocessors=[aap, iap])

    X_test = sdl_nivel.preprocess(image)
    X_test = X_test.astype("float") / 255.0

    return X_test

def run_predictions(image):

    y1_predict_index = nivel1_predict(image)
    y2_predict_index = nivel2_predict(image, y1_predict_index)
    y3_predict_index = nivel3_predict(image, y1_predict_index, y2_predict_index)
    
    label_nivel1 = class_hierarchy["produto"][y1_predict_index]
    label_nivel2 = class_hierarchy[label_nivel1][y2_predict_index]
    label_nivel3 = class_hierarchy[label_nivel2][y3_predict_index]

    predictions = [label_nivel1, label_nivel2, label_nivel3]
    
    return predictions


@app.route("/predict", methods=["POST"])
def predict():
    
    data = {"success": False}
    imageb64 = flask.request.json['image']

    image = Image.open(BytesIO(base64.b64decode(imageb64)))
    image = image.convert(mode='RGB')
    image = np.array(image)
    image = image[...,::-1]
    
    
    image = processing_image(image)
    # get detected product
    predictions = run_predictions(image)
    
    data["success"] = True
    data["Prediction Nivel 1"] = predictions[0]
    data["Prediction Nivel 2"] = predictions[1]
    data["Prediction Nivel 3"] = predictions[2]

    print(data)
    return flask.jsonify(data)


if __name__ == "__main__":
    load_all_models()
    app.run()