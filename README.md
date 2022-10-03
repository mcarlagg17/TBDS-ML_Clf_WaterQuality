# Calidad del Agua
Proyecto de **Machine Learning** cuyo objetivo es crear y comparar ***modelos de clasificación*** de la **calidad del agua**. Además, se implementa el modelo con mayor *precisión* en una *aplicación* *flask*
([repositorio Github de aplicación flask](https://github.com/mcarlagg17/TBDS_ML_Clf_WaterQuality_flask)).

![img](https://okdiario.com/img/2018/01/12/agua-cruda.jpg)

Se distingue entre ***segura y no segura*** a partir de los parámetros que se muestran a continuación:  



| Variable    | description                              |
|:------------|:-----------------------------------------|
| aluminium   | dangerous if greater than 2.8            |\n
| ammonia     | dangerous if greater than 32.5           |\n
| arsenic     | dangerous if greater than 0.01           |\n
| barium      | dangerous if greater than 2              |\n
| cadmium     | dangerous if greater than 0.005          |\n
| chloramine  | dangerous if greater than 4              |\n
| chromium    | dangerous if greater than 0.1            |\n
| copper      | dangerous if greater than 1.3            |\n
| flouride    | dangerous if greater than 1.5            |\n
| bacteria    | dangerous if greater than 0              |\n
| viruses     | dangerous if greater than 0              |\n
| lead        | dangerous if greater than 0.015          |\n
| nitrates    | dangerous if greater than 10             |\n
| nitrites    | dangerous if greater than 1              |\n
| mercury     | dangerous if greater than 0.002          |\n
| perchlorate | dangerous if greater than 56             |\n
| radium      | dangerous if greater than 5              |\n
| selenium    | dangerous if greater than 0.5            |\n
| silver      | dangerous if greater than 0.1            |\n
| uranium     | dangerous if greater than 0.3            |\n
| is_safe     | class attribute {0 - not safe, 1 - safe} |

## **Estructura** del proyecto 🗿 
- ***README.md***: *archivo actual, información inicial.*
- ***Proyecto_Machine_Learning.ipynb***: *notebook con el enunciado del proyecto.*
- ***proyect_resume.ipynb***: *notebook con la línea de desarrollo seguida en el proyecto. Se presentan las ideas, de los distintos notebooks que componen el proyecto, dando una visión general de los resultados obtenidos y cómo se ha llegado a ellos*

- ***src***:
    - **data**: carpeta con los dataset procesados y no procesados.
    - **img**: carpeta con figuras e imágenes utilizadas o extraídas del proyecto. 
    - **model**: 
        - modelos entrenados en archivos .pkl, ya preparados para su uso.
        - **metrics**: carpeta con los csv que continen las métricas extraídas de los distintos estimadores evaluados.
    - **utils**: carpeta donde se encuentran los archivos con las librerías necesarias y las funciones creadas.
    - *notebooks:* notebooks con el desarrollo del proyecto. A continuación se presenta el contenido de estos.

### *Contenido notebooks* 📌 
---
0. <a href='src/0_Introduccion.ipynb'>INTRODUCCIÓN</a>
1. ANÁLISIS EXPLORATORIO DE DATOS (<a href='src/1_EDA.ipynb'>EDA</a>)
    - 1.1. Acondicionamiento bases de datos
    - 1.2. Análisis visual
    - 1.3. Análisis estadístico   
2. MACHINE LEARNING (<a href='src/2a_ML_Baseline.ipynb'>*1era parte*</a> // <a href='src/2b_ML_BalancedData.ipynb'>*2a parte*</a>)
    - 2.1. Preparación y limpieza de datos
    - 2.2. Feature Engineering
    - 2.3. Modelado
3. <a href='src/3_Resultados.ipynb'>RESULTADOS</a>
    - 3.1. Visualización y reporting de los resultados
    - 3.2. Creación de un pipeline para el flujo automatizado
---

## Preparación 🔧

_Crear un entrono virtual y añadir las librerías mínimas para ejecutar este proyecto:_

* Creación del entorno:

```
>> conda create -n nombre_enviroment python==3.9.12

>> conda activate nombre_enviroment
```
* Instalar librerías:

Una vez creado el entorno, colocándonos en la carpeta *utils* dentro de *src* instalamos las librerías mínimas necesarias.

```
>> pip install -r requirements.txt
```

## Consejo de uso 🤓

Comenzar leyendo el archivo <a href='proyect_resume.ipynb'>proyect_resume</a> para entender mejor la organización y extensión de este proyecto. 

## Autora 👩🏽‍💻

* **María Carla González González** - [mcarlagg17](https://github.com/mcarlagg17)

### *Información de contacto:*
___
* Email: ***carla.glezz@gmail.com***
* Linkedin: ***https://www.linkedin.com/in/mariacarlagonzalezgonzalez/***
---

## Tutores 👨‍🏫

* **Marco Russo** - [marcusRB](https://github.com/marcusRB) 
* **Daniel Montes** - [DanielMontes](https://linkedin.com/in/daniel-montes-serrano-a81b9447)
* **Juan Maniglia** - [JuanManiglia](https://github.com/JuanManiglia)


## Agradecimientos 🤗

> A todxs los que de una forma u otra habéis formado parte de este viaje. Deseo lo mejor para cada uno de vosotrxs. 
 
> *¡Gracias de corazón!*




---
***Proyecto final Bootcamp Data Science***

---

![img](./src/img/logo.jpg)
