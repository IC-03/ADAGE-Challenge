# ⭐ sPCA Challenge
Repositorio del Challenge de reproducibilidad para análisis de datos a gran escala 2025-1

> 📄 **Citar el paper**: Elgamal, T., Yabandeh, M., Aboulnaga, A., Mustafa, W., & Hefeeda, M. (2015, May). sPCA: Scalable principal component analysis for big data on distributed platforms. In Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data (pp. 79-91).

> ✍️ **Descripción del paper**: 
El paper sPCA: Scalable Principal Component Analysis for Big Data on Distributed Platforms presenta un algoritmo optimizado para realizar Análisis de Componentes Principales (PCA) en entornos distribuidos, diseñado para manejar grandes volúmenes de datos. Los autores identifican limitaciones en las implementaciones existentes de PCA (como las de Mahout y MLlib) en términos de escalabilidad, precisión y generación de datos intermedios.

## 📂 Estructura del Proyecto
Para la ejecición del código solo usamos los archivos en `input` y el código `run_spca.py`, el resto de los archivos nos sirvieron para hacer pruebas del funcionamiento de la implementación.
```
/challenge
│
├── README.md             # Información sobre el proyecto 📚
├── .gitignore            # Archivos ignorados por Git 🚫
├── requirements.txt      # Dependencias del proyecto 📦
├── /src                  # Intento de un proyecto modular 🔧
├── /notebooks            # Notebooks para pruebas 📝
├── /input                # Datos de entrada en formato txt 📊
├── /PPCA.py              # Implementación naive del PCA 🔬
└── /run_spca.py          # Script principal para ejecutar SPCA en Spark 🚀
```

---

## 🚀 Correr el Código en Local

### 1. Clonar el repositorio
```bash
git clone <URL-del-repositorio>
cd challenge
```

### 2. Instalar las dependencias
Instalar las dependencias necesarias con `pip`:
```bash
pip install -r requirements.txt
```

### 3. Ejecutar el código
Correr el script con el siguiente comando, indicando el archivo de entrada, la dimensión de salida, las iteraciones máximas y la carpeta de salida:

```bash
python run_spca.py --input "./input/datos_1.txt" --dim 2 --maxIters 100 --output "./output/datos_1_pca_spark"
```

Donde:

* `--input` ✨: **Ruta** al archivo de entrada que contiene los datos que se desean reducir dimensionalmente.
* `--dim` 🎯: Número de dimensiones de salida que deseas obtener.
* `--maxIters` 🔄: Número máximo de iteraciones para el algoritmo en caso de que no converja.
* `--output` 💾: **Ruta** de la carpeta donde se guardarán los componentes principales y los datos con la dimensionalidad reducida.

---

## 🖥 Correr el Código en Cluster con Hadoop y Spark
Para correr el código en hadoop es clave subir los archivos de input a hdfs y pasarle el entorno virtual en un paquete para que los demás computadores del cluster puedan usar las librerías.

### 1. Clonar el repositorio en el cluster
```bash
git clone <URL-del-repositorio>
```

### 2. Crear y activar un entorno virtual
Crear un entorno virtual con `venv` y actívarlo:

```bash
python3 -m venv env
source env/bin/activate  # Para Linux/MacOS
# .\env\Scripts\activate  # Para Windows
```

### 3. Instalar las dependencias
Instalar las dependencias del proyecto:
```bash
pip install -r requirements.txt
```

### 4. Comprimir el entorno virtual
Empaquetar el entorno virtual en un archivo `env.tar.gz`:

```bash
tar -czvf env.tar.gz env/
```

### 5. Configuración de Spark en `run_spca.py`
Descomentar y configurar las siguientes líneas en `run_spca.py` para que use el entorno virtual comprimido en el cluster:

```python
############# Start Spark Session #############################

# For client mode
# spark = SparkSession.builder.appName("sPCA").getOrCreate()

# For cluster mode only
os.environ['PYSPARK_PYTHON'] = "./environment/env/bin/python"
spark = SparkSession.builder.config(
    "spark.yarn.dist.archives", "env.tar.gz#environment"
).appName("sPCA").getOrCreate()
```

### 6. Subir los archivos al HDFS
Subir los archivos de entrada al sistema HDFS de Hadoop y crea las carpetas necesarias:

```bash
hdfs dfs -put ./input/* /grupoh/challenge/input
hdfs dfs -mkdir /grupoh/challenge/output
```

### 7. Rezar 🙏

### 8. Ejecutar el código en el cluster
Ejecutar el código en el cluster de Spark con el siguiente comando:

```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --archives env.tar.gz#environment \
  --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/env/bin/python \
  --conf spark.executorEnv.PYSPARK_PYTHON=./environment/env/bin/python \
  run_spca.py \
  --input "hdfs:///grupoh/challenge/input/datos_1.txt" \
  --dim 2 \
  --maxIters 100 \
  --output "hdfs:///grupoh/challenge/output/datos_1_spark_pca"
```

### 9. Verificación de salida
Verificar que los archivos de entrada y salida se hayan cargado correctamente en las carpetas del HDFS. Al finalizar la ejecución, se  pueden encontrar los siguientes archivos en el directorio de salida:

```
/output
│
├── /datos_1_pca_spark_reduced  # Los datos con la dimensionalidad reducida
├── /datos_1_pca_spark_C        # Componentes principales (la matriz C)
```
