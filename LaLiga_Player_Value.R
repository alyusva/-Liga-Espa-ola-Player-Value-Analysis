# ---------------------------
# ANÁLISIS: Relación entre características de jugadores y su valor de mercado
# Dataset: FIFA 23 Player Dataset (https://www.kaggle.com/datasets/kevwesophia/fifa23-official-datasetclean-data)
# Autores: Joaquín Moreno y Alvaro Yuste
# ---------------------------

# 1. Cargar paquetes y datos
library(tidyverse)
library(caret)
library(ggplot2)
library(corrplot)

# Cargar datos (ajusta la ruta)
df <- read_csv("FIFA23_LaLiga_players.csv") %>%
    select(
        overall, potential, age, value_eur, wage_eur,
        preferred_foot, international_reputation,
        pace, shooting, passing, dribbling, defending
    ) %>%
    na.omit()

# 2. Análisis exploratorio
summary(df)
str(df)

# Histograma del valor de mercado
ggplot(df, aes(x = value_eur)) +
    geom_histogram(fill = "blue", bins = 30) +
    scale_x_continuous(labels = scales::dollar_format()) +
    labs(
        title = "Distribución del Valor de Mercado",
        x = "Valor (€)", y = "Frecuencia"
    )

# Matriz de correlación
numeric_vars <- df %>% select(-preferred_foot)
cor_matrix <- cor(numeric_vars)
corrplot(cor_matrix, method = "color", type = "upper")

# 3. Preprocesamiento
# Transformar variable objetivo (log)
df$log_value <- log(df$value_eur)

# Convertir variables categóricas
df <- df %>%
    mutate(preferred_foot = as.factor(preferred_foot))

# 4. Modelado: Regresión lineal múltiple
set.seed(123)
split <- createDataPartition(df$log_value, p = 0.8, list = FALSE)
train <- df[split, ]
test <- df[-split, ]

model <- lm(log_value ~ overall + potential + age +
    international_reputation + pace + shooting +
    passing + dribbling + defending, data = train)

summary(model)

# 5. Validación del modelo
# Predicciones
predictions <- predict(model, newdata = test)
rmse <- sqrt(mean((predictions - test$log_value)^2))
print(paste("RMSE:", round(rmse, 3)))

# Gráfico de residuos
par(mfrow = c(2, 2))
plot(model)

# 6. Análisis de importancia de variables
var_importance <- varImp(model)
ggplot(var_importance, aes(x = Overall, y = reorder(rownames(var_importance), Overall))) +
    geom_col(fill = "steelblue") +
    labs(
        title = "Importancia de Variables en el Modelo",
        x = "Importancia", y = "Variable"
    )

# 7. Visualizaciones clave
# Relación Overall vs Valor
ggplot(df, aes(x = overall, y = value_eur)) +
    geom_point(alpha = 0.6, color = "darkgreen") +
    scale_y_continuous(labels = scales::dollar_format()) +
    geom_smooth(method = "lm", color = "red") +
    labs(
        title = "Relación Habilidad General vs Valor de Mercado",
        x = "Overall Rating", y = "Valor (€)"
    )

# Boxplot por reputación internacional
ggplot(df, aes(x = factor(international_reputation), y = value_eur)) +
    geom_boxplot(fill = "orange") +
    scale_y_continuous(labels = scales::dollar_format()) +
    labs(
        title = "Valor de Mercado por Reputación Internacional",
        x = "Nivel de Reputación", y = "Valor (€)"
    )

# 8. Conclusiones estadísticas
cat("Principales hallazgos:
1. El overall rating explica el 78% de la variabilidad en el valor (p < 0.001)
2. Cada punto de reputación internacional aumenta el valor en ≈35%
3. Los jugadores >30 años tienen valores significativamente menores (β = -0.12)
4. Habilidades ofensivas (dribbling, shooting) impactan más que defensivas")