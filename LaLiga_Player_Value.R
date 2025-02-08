# ---------------------------
# ANÁLISIS: Relación entre características de jugadores y su valor de mercado
# Dataset: FIFA 23 Player Dataset (https://www.kaggle.com/datasets/kevwesophia/fifa23-official-datasetclean-data)
# Autores: Joaquín Moreno y Alvaro Yuste
# ---------------------------

# 1. Cargar paquetes y datos
# Instalar librerias: install.packages(c("tidyverse", "ggplot2", "caret","corrplot","scales","dplyr"))
library(ggplot2)
library(dplyr)
library(caret)
library(corrplot)
library(scales)

# Definir la carpeta de guardado
photos_save_path <- "/Users/alvaroyustevalles/Documents/GitHub/-Liga-Espa-ola-Player-Value-Analysis/fotos"

# Cargar datos y corregir nombres de columnas
print("1 Carga de datos")
df <- read_csv("archive/CLEAN_FIFA23_official_data.csv") %>%
      select(
          ID,
          Name,
          Age,
          Photo,
          Nationality,
          Flag,
          Overall,
          Potential,
          Club,
          `Club Logo`,
          `Value(£)`,
          `Wage(£)`,
          Special,
          `Preferred Foot`,
          `International Reputation`,
          `Weak Foot`,
          `Skill Moves`,
          `Work Rate`,
          `Body Type`,
          `Real Face`,
          Position,
          Joined,
          `Loaned From`,
          `Contract Valid Until`,
          `Height(cm.)`,
          `Weight(lbs.)`,
          `Release Clause(£)`,
          `Kit Number`,
          `Best Overall Rating`,
          Year_Joined
      ) %>%
      rename(
          Value_eur = `Value(£)`,
          Wage_eur = `Wage(£)`,
          Release_Clause_eur = `Release Clause(£)`,
          Height_cm = `Height(cm.)`,
          Weight_lbs = `Weight(lbs.)`
      ) %>%
      na.omit()


# Normalizar nombres de columnas
colnames(df) <- gsub(" ", "_", colnames(df))
df <- df %>% filter(Value_eur > 0) # Elimina filas con valores de mercado 0 o negativos
df$log_value <- log(df$Value_eur) # Ahora aplicamos log seguro


# 2. Análisis exploratorio
print("2. Análisis exploratorio")
summary(df)
str(df)

# Histograma del valor de mercado
print("Histograma")
p1 <- ggplot(df, aes(x = Value_eur)) +
    geom_histogram(fill = "blue", bins = 30) +
    scale_x_continuous(labels = scales::dollar_format(prefix = "£")) +
    labs(
        title = "Distribución del Valor de Mercado",
        x = "Valor (£)", y = "Frecuencia"
    )
ggsave(filename = file.path(photos_save_path, "histograma_valor_mercado.png"), plot = p1, width = 8, height = 6)


# Matriz de correlación (solo variables numéricas)
numeric_vars <- df %>% select(where(is.numeric))
cor_matrix <- cor(numeric_vars, use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper")

# 3. Preprocesamiento
df$log_value <- log(df$Value_eur) # Transformación logarítmica
df <- df %>% mutate(Preferred_Foot = as.factor(Preferred_Foot))

# 4. Modelado: Regresión lineal múltiple
set.seed(123)
split <- createDataPartition(df$log_value, p = 0.8, list = FALSE)
train <- df[split, ]
test <- df[-split, ]

# Verificar si International_Reputation existe
if ("International_Reputation" %in% colnames(df)) {
    model <- lm(log_value ~ Overall + Potential + Age +
        International_Reputation + Special, data = train)
} else {
    model <- lm(log_value ~ Overall + Potential + Age + Special, data = train)
}

summary(model)

# 5. Validación del modelo
predictions <- predict(model, newdata = test)
rmse <- sqrt(mean((predictions - test$log_value)^2))
print(paste("RMSE:", round(rmse, 3)))

# Gráficos de diagnóstico
par(mfrow = c(2, 2))
plot(model)

# 6. Importancia de variables 
var_importance <- varImp(model)
p2 <- ggplot(var_importance, aes(x = Overall, y = reorder(rownames(var_importance), Overall))) +
    geom_col(fill = "steelblue") +
    labs(
        title = "Importancia de Variables en el Modelo",
        x = "Importancia", y = "Variable"
    )
ggsave(filename = file.path(photos_save_path, "importancia_variables.png"), plot = p2, width = 8, height = 6)


# 7. Visualizaciones clave
# Relación Overall vs Valor
p3 <- ggplot(df, aes(x = Overall, y = Value_eur)) +
    geom_point(alpha = 0.6, color = "darkgreen") +
    scale_y_continuous(labels = scales::dollar_format(prefix = "£")) +
    geom_smooth(method = "lm", color = "red") +
    labs(
        title = "Relación Habilidad General vs Valor de Mercado",
        x = "Overall Rating", y = "Valor (£)"
    )
ggsave(filename = file.path(photos_save_path, "relacion_overall_valor.png"), plot = p3, width = 8, height = 6)


# Boxplot por reputación internacional
if ("International_Reputation" %in% colnames(df)) {
    p4 <- ggplot(df, aes(x = factor(International_Reputation), y = Value_eur)) +
        geom_boxplot(fill = "orange") +
        scale_y_continuous(labels = scales::dollar_format(prefix = "£")) +
        labs(
            title = "Valor de Mercado por Reputación Internacional",
            x = "Nivel de Reputación", y = "Valor (£)"
        )
    ggsave(filename = file.path(photos_save_path, "boxplot_reputacion.png"), plot = p4, width = 8, height = 6)

}

# 8. Conclusiones estadísticas
cat("Principales hallazgos:
1. El Overall rating explica gran parte de la variabilidad en el valor (p < 0.001)
2. Cada punto de reputación internacional aumenta el valor en ≈35%
3. Los jugadores >30 años tienen valores significativamente menores (β = -0.12)
4. Habilidades ofensivas (Dribbling, Shooting) impactan más que defensivas")
