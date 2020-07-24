# APIFoodClassification
Modelo que prevê o nome e o tipo de um produto numa foto.

## Instalação

- Clonamos o repositorio ``https://github.com/yulenn/APIFoodClassification``
- Instalamos pip3 ``sudo apt install python3-pip``
- Instalamos Python3.7 https://linuxize.com/post/how-to-install-python-3-7-on-ubuntu-18-04/

Utilizamos [Pipenv](https://pipenv-fork.readthedocs.io/en/latest/basics.html) para criar um ambiente com as dependencias do projeto 

- Instalamos o ``Pipenv`` com ``pip3 install pipenv``

E instalamos todas as dependencias do projeto que estão no arquivo ``Pipfile`` com 
- o comando ``pipenv install``
Agora temos as bibliotecas que precisamos instaladas no ambiente do projeto.

- Depois criamos uma pasta ``models`` no repositorio principal e entramos nela com ``cd models``

- Colocamos os modelos que estão no seguinte endereço [Modelos - Drive](https://drive.google.com/drive/folders/1NIPUqkbaWMSTtiH_ImTR5raEnJzAyTBB?usp=sharing) dentro do diretorio ``models``

Pronto =)

## Execução

- Ativamos nosso ambiente do projeto com nossas dependencias instalas com o comando ``pipenv shell``

- Rodamos o script ``run_keras_server.py`` com ``python run_keras_server.py``
Ele vai ligar nosso servidor que escuta o porto ``http://127.0.0.1:5000/``


## Exemplo de uso
Utilizamos Postman como cliente que envia uma requisição POST no seguinte endereço ``http://127.0.0.1:5000/predict``

- Nossa imagem query é

  ![](https://github.com/yulenn/APIFoodClassification/blob/master/test_images/coca-original.jpg)

- Porem precisamos obter o formato ``base64`` da imagem query que é:
  [encoding BASE64 imagem query](https://github.com/yulenn/APIFoodClassification/blob/master/test_images/base64_cocacola-original.txt)

- Aqui temos um panorama geral da consulta no Postman

  ``
    POST: / 
    body: {
    image: IMAGE_B64_AS_STRING
    }
  ``

  ![](https://github.com/yulenn/APIFoodClassification/blob/master/test_images/resultado.png)

- Apertamos o butão SEND e obtemos a resposta do servidor

  ![](https://github.com/yulenn/APIFoodClassification/blob/master/test_images/labels_detetatos.png)


## Lista de Produtos considerados no projeto
Foram coletadas as fotos desses produtos com o telefone, para o treinamento dos modelos na data do Julio 2019 até o Julio 2020.

``
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
``
