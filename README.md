# APIFoodClassification
Deteta qual o produto na imagem capturada por um telefone

## Installation
Estamos utilizando o [Pipenv](https://pipenv-fork.readthedocs.io/en/latest/basics.html) para salvar as dependencias do projeto. Para instalar as dependencias fazemos o seguinte:
- Primeiro temos que ter o pipenv instalado utilizando ``pip install pipenv``
- Depois clonar este repositorio ``https://github.com/yulenn/APIFoodClassification.git`` na maquina local.
- Instalar as dependencias que estão listadas no arquivo``Pipfile`` utilizando ``pipenv install``.
Pronto, voce tem as dependencias instaladas no environment do projeto.

Agora, temos que adicionar os modelos de deteção dos produtos das imagens, para isso:
- Criar um diretorio com o nome ``models`` no repositorio local
- Entrar nesse diretorio ``cd models`` e dentro dele colocar os seguintes arquivos que estão no seguinte enlace no Drive:
  [Modelos de classificação - Drive](https://drive.google.com/drive/folders/1NIPUqkbaWMSTtiH_ImTR5raEnJzAyTBB?usp=sharing)

Pronto =)

## Run
- Retornar no repositorio principal ``APIProductClassification``
- Ativar o ``Pipenv`` com o comando ``pipenv shell`` para entrar no enviroment do projeto
- Por fim, rodar o script ``run_keras_server.py`` com ``python run_keras_server.py``
  Com isso, o servidor está ouvindo o seguinte endpoint ``http://127.0.0.1:5000``

## Exemplo de uso
- Imagem de consulta
  ![imagem consulta](https://github.com/yulenn/APIFoodClassification/blob/master/test_images/suco_dobem_limonada.jpg)
- Na verdade, precisamos o formato ``base64`` da imagem consulta que é:
  ![base64](https://github.com/yulenn/APIFoodClassification/blob/master/test_images/base64_suco_dobem_limonada.txt) 

- Utilizamos o postman como cliente para enviar uma requisição, para o ``http://127.0.0.1:5000/predict`` com o formato ``base64`` da imagem de consulta, e apertamos no botão ``Send``, veja-se:
![resultado](https://github.com/yulenn/APIFoodClassification/blob/master/test_images/resultado.png)

- Obtemos os labels detetados da imagem query
![Rótulos detetados](https://github.com/yulenn/APIFoodClassification/blob/master/test_images/labels_detetatos.png)

