# Proyecto Integrador I: Análisis de la relación entre datos con Algoritmia
Las matemáticas que hacen posible el ‘Data Science’

PALABRAS CLAVE: Aprendizaje Automático, Machine Learning, Ciencia de Datos, Análisis de Componentes Principales (PCA), K-means Clustering, Estadística Socioeconómica.



## Requisitos previos

- Disponer de [Python 3.8+](https://www.python.org/downloads/) instalado en el sistema.
- Disponer de [Visual Studio Code](https://code.visualstudio.com/) instalado.

## 1. Instalación de extensiones necesarias en Visual Studio Code

Se recomienda instalar las siguientes extensiones desde el marketplace de Visual Studio Code:

- **Python** (Microsoft)
- **Jupyter** (Microsoft)

## 2. Creación y activación de un entorno virtual en Python

Desde una terminal ubicada en la carpeta raíz del proyecto, ejecutar:

```bash
python -m venv venv
```

Para activar el entorno virtual, utilizar el comando correspondiente según el sistema operativo:

- En Windows:
  ```bash
  .\venv\Scripts\activate
  ```
- En macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

## 3. Instalación del kernel de Jupyter en el entorno virtual

Con el entorno virtual activado, instalar Jupyter y el kernel ejecutando:

```bash
pip install notebook ipykernel
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```

## 4. Instalación de las librerías necesarias para el proyecto

Asegurarse de que el entorno virtual esté activo y ejecutar:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 5. Apertura y ejecución del notebook

1. Abrir Visual Studio Code en la carpeta del proyecto.
2. Abrir el archivo `Python Version.ipynb`.
3. Seleccionar el kernel "Python (venv)" en la parte superior derecha del notebook.
4. Ejecutar las celdas del notebook según corresponda.

---

**Nota:**  
Es imprescindible que los archivos de datos (`Monografia_final.csv`, `FINAL_DATOS_IMPUTADOS-2.csv`, etc.) se encuentren en la carpeta `datasets` dentro del directorio del proyecto.
