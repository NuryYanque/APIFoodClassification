# APIFoodClassification
Modelo que prevê o nome e o tipo de um produto numa foto.

## Instalação

- Clonamos o repositorio ``https://github.com/yulenn/APIFoodClassification``

Utilizamos [Pipenv](https://pipenv-fork.readthedocs.io/en/latest/basics.html) para criar um ambiente com as dependencias do projeto 

- Instalamos o ``Pipenv`` com ``pip install pipenv``

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
  ![](https://github.com/yulenn/APIFoodClassification/blob/master/test_images/resultado.png)

- Apertamos o butão SEND e obtemos a resposta do servidor
  ![](https://github.com/yulenn/APIFoodClassification/blob/master/test_images/labels_detetatos.png)
