# LaLiga_Players_Value_Analysis
Practica Final Estadística Avanzada  - Master AI &amp; Big Data

# 🚀 Liga Española Player Value Analysis⚽️

Análisis estadístico avanzado para determinar la relación entre características de jugadores de fútbol y su valor de mercado, utilizando datos de la Liga Española (FIFA 23).

---
## 📋 Contenido del Proyecto
- **Código R**: Script completo (`LaLiga_Player_Value.R`) que incluye el análisis exploratorio, modelado (Regresión Lineal y Random Forest) y visualizaciones.
- **Dataset**: Datos limpios de jugadores de FIFA 23, extraídos del [FIFA 23 Player Dataset (Kaggle)](https://www.kaggle.com/datasets/kevwesophia/fifa23-official-datasetclean-data).
- **Informe Final PDF**: Documento con conclusiones detalladas, hallazgos, comparaciones de modelos y análisis estadístico.
- **Gráficos**: Visualizaciones generadas durante el análisis, disponibles en la carpeta `results/graficos`.

---
## ⚙️ Configuración

### Requisitos

- **R** (v4.3.1 o superior)
- **RStudio** (recomendado) o VS Code con extensión R.
- **Paquetes de R**:
  ```r
  install.packages(c("tidyverse", "ggplot2", "caret", "corrplot", "scales", "dplyr", "psych", "randomForest"))
  ```
### Instrucciones para Ejecutar el Proyecto
**Descargar el dataset de Kaggle:**
```r
kaggle datasets download -d kevwesophia/fifa23-official-datasetclean-data
```
Coloca el archivo CLEAN_FIFA23_official_data.csv en la carpeta /datasets.
**Clonar el repositorio:**
```r
git clone https://github.com/tuusuario/LaLiga-PlayerValue-FIFA23-Analysis.git
```
**Ejecutar el script LaLiga_Player_Value.R en R/RStudio.**

### 📊 Resultados Clave
- **Variables clave en el valor de mercado:**
Overall: Correlación con log(valor) r = 0.54.
International Reputation: r = 0.23.
Potential y Skill Moves: r = 0.52 y r = 0.31, respectivamente.
En Random Forest, al incluir la variable Release_Clause_eur, ésta emerge como la más influyente, lo que refleja su fuerte correlación con el valor de mercado.
En Regresión Lineal, se observa que tanto Overall como Release_Clause_eur presentan coeficientes significativos y altos valores de t cuando se incluyen todas las variables.

- **Cuantificación de Habilidades Técnicas:**
El modelo de regresión lineal simple para Overall explica aproximadamente el 29% de la variabilidad del log(valor) (R² = 0.29).
Cada punto adicional en Overall incrementa el log(valor) en ~0.14 unidades.

- **Modelado Predictivo:**
Regresión Lineal Múltiple: R² ajustado = 0.45 en entrenamiento; RMSE en test = 1.443.
Random Forest: RMSE en test = 1.227, lo que sugiere mayor robustez frente a outliers y una mejor captura de la complejidad de los datos.

- **Comparación de Modelos::**
El Random Forest presenta un RMSE inferior, evidenciando un desempeño predictivo superior.
La Regresión Lineal aporta mayor interpretabilidad a través de sus coeficientes, facilitando la comprensión de la relación entre variables y el valor de mercado.

### 🗂️ Estructura del Proyecto
```
├── datasets/
│   └── CLEAN_FIFA23_official_data.csv  # Dataset original
├── scripts/
│   └── LaLiga_Player_Value.R           # Código de análisis
├── results/
│   ├── informe_final.pdf               # Informe detallado
│   └── graficos/                       # Gráficos generados
└── README.md
```

### 📄 Licencia
Este proyecto se distribuye bajo la licencia MIT.

### ✉️ Contacto
- **Joaquín Moreno: juaki1502@gmail.com**
- **Álvaro Yuste: alyusva@gmail.com**

**🔗 Enlace al Dataset: FIFA 23 Player Dataset**
