# LaLiga_Players_Value_Analysis
Practica Final Estadística Avanzada  - Master AI &amp; Big Data

# 🚀 Liga Española Player Value Analysis⚽️

Análisis estadístico avanzado para determinar la relación entre características de jugadores de fútbol y su valor de mercado, utilizando datos de la Liga Española (FIFA 23).

---
## 📋 Contenido del Proyecto
- **Código R**: Script completo con análisis exploratorio, modelado y visualizaciones.
- **Dataset**: Datos limpios de jugadores de la Liga Española (FIFA 23).
- **Informe PDF**: Conclusiones detalladas y hallazgos estadísticos.
---
## ⚙️ Configuración

### Requisitos

- **R** (v4.3.1 o superior)
- **RStudio** (recomendado) o VS Code con extensión R.
- **Paquetes de R**:
  ```r
  install.packages(c("tidyverse", "caret", "ggplot2", "corrplot"))
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
Overall Rating explica el 78% de la variabilidad (R² = 0.87).
Cada nivel de reputación internacional aumenta el valor en un 35%.
Edad >30 años reduce el valor significativamente (β = -0.12).

- **Visualizaciones:**
Correlación entre habilidades y valor.
Distribución del valor de mercado por reputación internacional.
Modelo predictivo con RMSE = 0.243 (escala logarítmica).

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
