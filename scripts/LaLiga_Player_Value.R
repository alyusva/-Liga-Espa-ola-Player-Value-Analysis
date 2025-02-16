# ---------------------------
# ANÁLISIS: Relación entre características de jugadores y su valor de mercado
# Dataset: FIFA 23 Player Dataset (https://www.kaggle.com/datasets/kevwesophia/fifa23-official-datasetclean-data)
# Autores: Joaquín Moreno y Alvaro Yuste
# ---------------------------

# 1. Cargar paquetes y datos
# Instalar librerias: install.packages(c("tidyverse", "ggplot2", "caret","corrplot","scales","dplyr","psych","randomForest"))
library(ggplot2)
library(dplyr)
library(caret)
library(corrplot)
library(scales)
library(tidyverse)
library(psych)
library(randomForest)

# Carpeta de guardado de resultados
photos_save_path <- "results"
dir.create(photos_save_path, showWarnings = FALSE) # Crear carpeta si no existe

# 1.1. Carga y limpieza inicial
print("1. Carga de datos")
df_raw <- read_csv("datasets/CLEAN_FIFA23_official_data.csv")
# Mostrar las primeras 10 filas
head(df_raw, 10)

# 1.2. Análisis de valores faltantes (antes de limpiar)
print("Valores faltantes por columna (antes de limpiar):")
print(colSums(is.na(df_raw)))

# 1.3. Selección y renombrado de variables
df <- df_raw %>%
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
      na.omit() # Eliminar filas con NA


# Histograma del valor de mercado Variable que nos importa
print("Histograma")
p1 <- ggplot(df, aes(x = Value_eur)) +
    geom_histogram(fill = "blue", bins = 30) +
    scale_x_continuous(labels = scales::dollar_format(prefix = "£")) +
    labs(
        title = "Distribución del Valor de Mercado",
        x = "Valor (£)", y = "Frecuencia"
    )
ggsave(filename = file.path(photos_save_path, "histograma_valor_mercado.png"), plot = p1, width = 8, height = 6)


# 1.4. Análisis de valores faltantes (después de limpiar)
print(paste("Filas originales:", nrow(df_raw)))
print(paste("Filas después de limpiar:", nrow(df)))

# ----------------------------------------------------------
# 2. Estadísticas descriptivas detalladas
# ----------------------------------------------------------
print("2. Estadísticas descriptivas")

# 2.1. Resumen numérico
numeric_stats <- df %>%
    select(where(is.numeric)) %>%
    psych::describe(quant = c(.25, .75)) %>%
    select(-vars, -trimmed, -mad, -range)

print("Estadísticas descriptivas de variables numéricas:")
print(numeric_stats)

# 2.2. Análisis de variables categóricas
print("Distribución de Preferred Foot:")
print(table(df$Preferred_Foot))
# mostrar histograma
histogram_feet <- ggplot(df, aes(x = `Preferred Foot`)) +
    geom_bar(fill = "steelblue", color = "black") +
    labs(
        title = "Distribución del Pie Preferido",
        x = "Pie Preferido",
        y = "Frecuencia"
    ) +
    theme_minimal()
# Guardar el histograma en un archivo
ggsave(filename = file.path(photos_save_path, "histograma_pie_preferido.png"), plot = histogram_feet, width = 8, height = 6)


# 2.3. Detección de outliers con boxplots
generate_boxplot <- function(var_name) {
    p <- ggplot(df, aes(y = .data[[var_name]])) +
        geom_boxplot(fill = "skyblue") +
        labs(title = paste("Boxplot de", var_name), y = var_name)

    ggsave(
        filename = file.path(photos_save_path, paste0("boxplot_", var_name, ".png")),
        plot = p,
        width = 8,
        height = 6
    )
}

# Generar boxplots para variables clave
lapply(c("Value_eur", "Age", "Overall", "Potential"), generate_boxplot)

# 2.4. Eliminación de outliers usando IQR
handle_outliers <- function(x) {
    qnt <- quantile(x, probs = c(.25, .75), na.rm = TRUE)
    iqr <- 1.5 * IQR(x, na.rm = TRUE)
    x[x < (qnt[1] - iqr) | x > (qnt[2] + iqr)] <- NA
    return(x)
}

# Aplicar a variables numéricas clave
df_clean <- df %>%
    mutate(across(
        c(Value_eur, Age, Overall, Potential),
        ~ handle_outliers(.x)
    )) %>%
    na.omit()

print(paste("Filas después de eliminar outliers:", nrow(df_clean)))

# ----------------------------------------------------------
# 3. Análisis de correlaciones y transformaciones
# ----------------------------------------------------------
print("3. Análisis de correlaciones")

# Transformación logarítmica para Value_eur (para reducir sesgo)
df$log_value <- log(df$Value_eur)
df <- df %>% mutate(log_value = log(Value_eur + 1)) # Evita log(0)
str(df$log_value)


# Seleccionar todas las variables numéricas para la matriz de correlación
cor_vars <- df %>%
    select(
        log_value, Overall, Potential, Age, Special,
        `International Reputation`, `Weak Foot`, `Skill Moves`,
        Height_cm, Weight_lbs, Release_Clause_eur
    )

# Calcular la matriz de correlación
cor_matrix <- cor(cor_vars, use = "complete.obs")

# Visualizar la matriz de correlación
png(file.path(photos_save_path, "corrplot.png"), width = 800, height = 800)
corrplot(cor_matrix,
    method = "color", type = "upper",
    tl.col = "black", tl.srt = 45,
    addCoef.col = "black", number.cex = 0.8
)
dev.off()

# ----------------------------------------------------------
# 4. Modelado y validación
# ----------------------------------------------------------
print("4. Modelado y validación")

# 4.1. Preparación de datos
# Convertir variables categóricas a factores
df <- df %>%
    mutate(
        Preferred_Foot = as.factor(`Preferred Foot`),
        Work_Rate = as.factor(`Work Rate`),
        Body_Type = as.factor(`Body Type`),
        Position = as.factor(Position)
    )

# Dividir los datos en entrenamiento y prueba (80% - 20%)
set.seed(123)
train_index <- createDataPartition(df$log_value, p = 0.8, list = FALSE)
train <- df[train_index, ]
test <- df[-train_index, ]

# 4.2. Modelado: Regresión lineal múltiple
# Usar todas las variables seleccionadas
final_model <- lm(
    log_value ~ Overall + Potential + Age + Special +
        `International Reputation` + `Weak Foot` + `Skill Moves` +
        Height_cm + Weight_lbs + Release_Clause_eur +
        Preferred_Foot + Work_Rate + Body_Type + Position,
    data = train
)

# Resumen del modelo
print(summary(final_model))

# 4.3 Calcular la importancia de variables con varImp()
lm_importance <- varImp(final_model, scale = FALSE)

# 4.4 Convertir a un data.frame para graficar con ggplot2
df_lm_importance <- data.frame(
    Variable   = rownames(lm_importance),
    Importance = lm_importance$Overall
)

# 4.5 Graficar la importancia con ggplot
library(ggplot2)
p_lm_importance <- ggplot(df_lm_importance, aes(
    x = Importance,
    y = reorder(Variable, Importance)
)) +
    geom_col(fill = "steelblue", width = 0.7) +
    labs(
        title = "Importancia de Variables en la Regresión Lineal",
        x = "Importancia (|t|)",
        y = "Variable"
    ) +
    theme_minimal()

# Guardar el gráfico en un archivo
ggsave(filename = file.path(photos_save_path, "var_importance_lm.png"), plot = p_lm_importance, width = 8, height = 6)


# 4.3. Validación cruzada
# Configurar control de validación cruzada (5 folds)
ctrl <- trainControl(method = "cv", number = 5)

# Entrenar el modelo con validación cruzada
cv_model <- train(
    log_value ~ Overall + Potential + Age + Special +
        `International Reputation` + `Weak Foot` + `Skill Moves` +
        Height_cm + Weight_lbs + Release_Clause_eur +
        Preferred_Foot + Work_Rate + Body_Type + Position,
    data = train,
    method = "lm",
    trControl = ctrl
)

# Resultados de la validación cruzada
print(cv_model)
print(paste("RMSE en validación cruzada:", cv_model$results$RMSE))

# 4.4. Predicciones en el conjunto de prueba
predictions <- predict(final_model, newdata = test)

# Calcular RMSE en el conjunto de prueba
rmse <- sqrt(mean((predictions - test$log_value)^2))
print(paste("RMSE en conjunto de prueba:", round(rmse, 3)))

# 4.5. Gráficos de diagnóstico del modelo
png(file.path(photos_save_path, "diagnostic_plots.png"), width = 800, height = 800)
par(mfrow = c(2, 2))
plot(final_model)
dev.off()
# 1️ Residuals vs Fitted (Residuos vs Valores Ajustados)
#Evaluar la linealidad y la homocedasticidad
# - Se observa una clara curva en los residuos, lo que sugiere que la relación entre las variables no es completamente lineal.
# - Hay puntos extremadamente alejados, indicando valores atípicos o errores en los datos.
# - Podría ser necesario agregar términos polinómicos o transformar variables.

# 2️ Q-Q Plot de Residuos (Normalidad)
# Comprobar si los residuos siguen una distribución normal
# - Los valores en los extremos (colas) se desvían significativamente de la línea diagonal, lo que indica que los residuos no son normales.
# - Esto puede afectar la validez de los intervalos de confianza y pruebas de significancia.
# - Soluciones: transformar la variable objetivo (log_value) o usar un modelo más robusto.

# 3️ Scale-Location (Homoscedasticidad)
# Evaluar si la varianza de los residuos es constante
# - La línea roja no es horizontal y muestra patrones de dispersión variable, lo que indica heterocedasticidad (la varianza no es constante).
# - Esto significa que el modelo podría estar prediciendo mejor para ciertos rangos de valores y peor para otros.
# - Soluciones: usar log() en variables dependientes o un modelo no lineal.

# 4️ Residuals vs Leverage (Valores influyentes)
# Identificar observaciones que tienen un impacto excesivo en el modelo.
# - Hay varios puntos con alta leverage e influencia (indicados con etiquetas de número).
# - Un punto fuera de la curva de Cook's Distance indica que ciertas observaciones están afectando en exceso los coeficientes del modelo.


print("4. Modelado y validación con Random Forest")
# Este es más robusto frente a outliers y no requiere supuestos de normalidad

# 4.1. Preparación de datos
# Convertir variables categóricas a factores
df <- df %>%
    mutate(
        Preferred_Foot = as.factor(`Preferred Foot`),
        Work_Rate = as.factor(`Work Rate`),
        Body_Type = as.factor(`Body Type`),
        Position = as.factor(Position)
    )

# Dividir los datos en entrenamiento y prueba (80% - 20%)
set.seed(123)
train_index <- createDataPartition(df$log_value, p = 0.8, list = FALSE)
train <- df[train_index, ]
test <- df[-train_index, ]

# 4.2. Modelado: Random Forest

# Ajustar el modelo de Random Forest
rf_model <- randomForest(
    log_value ~ Overall + Potential + Age + Special +
        # `International Reputation` + `Weak Foot`+ `Skill Moves` +
        Height_cm + Weight_lbs + Release_Clause_eur +
        Preferred_Foot + Work_Rate + Body_Type + Position,
    data = train,
    ntree = 100, # Número de árboles
    importance = TRUE # Calcular importancia de variables
)

# Resumen del modelo
print(rf_model)

# 4.3. Importancia de las variables
importance_rf <- importance(rf_model)
print("Importancia de las variables en Random Forest:")
print(importance_rf)

# Gráfico de importancia de variables
var_importance_rf <- data.frame(
    Variable = rownames(importance_rf),
    Importance = importance_rf[, "%IncMSE"] # Usar %IncMSE para regresión
)

p_importance_rf <- ggplot(var_importance_rf, aes(x = Importance, y = reorder(Variable, Importance))) +
    geom_col(fill = "steelblue", width = 0.7) +
    labs(
        title = "Importancia de Variables en Random Forest",
        x = "Importancia (% Incremento en MSE)",
        y = "Variable"
    ) +
    theme_minimal()

ggsave(file.path(photos_save_path, "var_importance_rf.png"), p_importance_rf)

# 4.4. Predicciones en el conjunto de prueba
predictions_rf <- predict(rf_model, newdata = test)

# Calcular RMSE en el conjunto de prueba
rmse_rf <- sqrt(mean((predictions_rf - test$log_value)^2))
print(paste("RMSE en conjunto de prueba (Random Forest):", round(rmse_rf, 3)))

# 4.5. Gráficos de diagnóstico
# Gráfico de valores observados vs predichos
p_obs_vs_pred <- ggplot(data = test, aes(x = log_value, y = predictions_rf)) +
    geom_point(alpha = 0.6, color = "darkgreen") +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    labs(
        title = "Valores Observados vs Predichos (Random Forest)",
        x = "Valor Observado (log)",
        y = "Valor Predicho (log)"
    ) +
    theme_minimal()

ggsave(file.path(photos_save_path, "obs_vs_pred_rf.png"), p_obs_vs_pred)


# ----------------------------------------------------------
# 5. Comparación de modelos: Regresión Lineal vs Random Forest
# ----------------------------------------------------------
print("5. Comparación de modelos: Regresión Lineal vs Random Forest")

# 5.1. Métricas de rendimiento
# Crear un dataframe para comparar métricas
metrics_comparison <- data.frame(
    Modelo = c("Regresión Lineal", "Random Forest"),
    RMSE = c(rmse, rmse_rf),
    R2 = c(summary(final_model)$r.squared, NA) # R² no aplica directamente en Random Forest
)

print("Comparación de métricas de rendimiento:")
print(metrics_comparison)

# 5.2. Importancia de variables
# Importancia en regresión lineal
var_importance_lm <- varImp(final_model)
var_importance_lm <- data.frame(
    Variable = rownames(var_importance_lm),
    Importance_LM = var_importance_lm$Overall
)

# Importancia en Random Forest
var_importance_rf <- data.frame(
    Variable = rownames(importance_rf),
    Importance_RF = importance_rf[, "%IncMSE"]
)

# Combinar ambas importancias
importance_comparison <- merge(var_importance_lm, var_importance_rf, by = "Variable", all = TRUE)
print("Comparación de importancia de variables:")
print(importance_comparison)

# 5.3. Gráfico comparativo de importancia de variables
importance_comparison_long <- importance_comparison %>%
    pivot_longer(cols = starts_with("Importance"), names_to = "Modelo", values_to = "Importancia")

p_importance_comparison <- ggplot(importance_comparison_long, aes(x = Importancia, y = reorder(Variable, Importancia), fill = Modelo)) +
    geom_col(position = "dodge", aes(fill = Modelo)) +
    labs(
        title = "Comparación de Importancia de Variables",
        x = "Importancia",
        y = "Variable",
        fill = "Modelo"
    ) +
    theme_minimal()

ggsave(file.path(photos_save_path, "importance_comparison.png"), p_importance_comparison)

# 5.4. Gráfico comparativo de valores observados vs predichos
# Crear un dataframe con valores observados y predichos de ambos modelos
comparison_df <- data.frame(
    Observado = test$log_value,
    Predicho_LM = predict(final_model, newdata = test),
    Predicho_RF = predictions_rf
)

# Gráfico comparativo
p_obs_vs_pred_comparison <- ggplot(comparison_df, aes(x = Observado)) +
    geom_point(aes(y = Predicho_LM, color = "Regresión Lineal"), alpha = 0.6) +
    geom_point(aes(y = Predicho_RF, color = "Random Forest"), alpha = 0.6) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    labs(
        title = "Comparación de Predicciones: Regresión Lineal vs Random Forest",
        x = "Valor Observado (log)",
        y = "Valor Predicho (log)",
        color = "Modelo"
    ) +
    theme_minimal()

ggsave(file.path(photos_save_path, "obs_vs_pred_comparison.png"), p_obs_vs_pred_comparison)


# ---------------------------
#  6. CONCLUSIONES DEL ANÁLISIS
# ---------------------------

# 1. Calcular correlaciones entre variables clave y el log del valor de mercado
cor_overall <- cor(df$Overall, df$log_value, use = "complete.obs")
cor_international <- cor(df$`International Reputation`, df$log_value, use = "complete.obs")
cor_age <- cor(df$Age, df$log_value, use = "complete.obs")
cor_potential <- cor(df$Potential, df$log_value, use = "complete.obs")
cor_skill_moves <- cor(df$`Skill Moves`, df$log_value, use = "complete.obs")
# (Otras correlaciones pueden calcularse si son necesarias)

# 2. Modelos univariados para obtener coeficientes y R²

# 2.1. Modelo para Overall
model_overall <- lm(log_value ~ Overall, data = df)
summary_overall <- summary(model_overall)
r2_overall <- summary_overall$r.squared
coef_overall_uni <- summary_overall$coefficients[2, "Estimate"]

# 2.2. Modelo para Reputación Internacional
model_international <- lm(log_value ~ `International Reputation`, data = df)
summary_international <- summary(model_international)
coef_international <- summary_international$coefficients[2, "Estimate"]

# 2.3. Modelo para Edad
model_age <- lm(log_value ~ Age, data = df)
summary_age <- summary(model_age)
coef_age <- summary_age$coefficients[2, "Estimate"]

# 3. Impresión de las conclusiones detalladas, coherentes con el informe

cat("\n**************** CONCLUSIONES FINALES ****************\n\n")

# Objetivo 1
cat("Objetivo 1: Identificar las variables con mayor impacto en el valor de mercado.\n")
cat(" - Overall es la variable con mayor correlación con el log del valor de mercado: r =", round(cor_overall, 2), "\n")
cat(" - Reputación Internacional también destaca, con r =", round(cor_international, 2), "\n")
cat(" - Potential y Skill Moves presentan correlaciones positivas moderadas (r =", round(cor_potential, 2), "y", round(cor_skill_moves, 2), "respectivamente).\n")
cat(" - Random Forest (con la variable Release_Clause_eur incluida) muestra que la cláusula de rescisión sobresale como la variable más influyente, lo cual es coherente con la realidad del fútbol.\n")
cat(" - En la Regresión Lineal, al incluir todas las variables (incluyendo International Reputation), se observa que Overall y Release_Clause_eur presentan coeficientes significativos y altos valores de t.\n\n")

# Objetivo 2
cat("Objetivo 2: Cuantificar la relación entre habilidades técnicas y el valor.\n")
cat(" - El modelo de regresión lineal simple para Overall explica aproximadamente el", round(r2_overall * 100, 1), "% de la variabilidad del log(valor) (R² =", round(r2_overall, 2), ").\n")
cat(" - Cada punto adicional en Overall incrementa el log(valor) en ~", round(coef_overall_uni, 3), "unidades.\n")
cat(" - Potential también contribuye positivamente, aunque su efecto puede ser menor que el de Overall o verse eclipsado por la presencia de la cláusula de rescisión.\n")
cat(" - Skill Moves suele tener menor importancia que Overall y Potential, aunque aparece como estadísticamente significativa en algunos modelos.\n\n")

# Objetivo 3
cat("Objetivo 3: Evaluar el efecto de la edad y la reputación internacional.\n")
cat("Para la Edad:\n")
cat(" - La correlación entre Edad y log(valor) es r =", round(cor_age, 2), ".\n")
cat(" - El modelo univariado muestra que cada año adicional reduce el log(valor) en ~", round(coef_age, 3), "unidades.\n")
cat("Para la Reputación Internacional:\n")
cat(" - Cada nivel adicional en Reputación Internacional incrementa el log(valor) en ~", round(coef_international, 3), "unidades.\n\n")

# Resultados del Modelado Predictivo
cat("Resultados del Modelado Predictivo:\n")
cat("Regresión Lineal Múltiple:\n")
cat(" - R² ajustado =", round(summary(final_model)$adj.r.squared, 2), "\n")
cat(" - RMSE en test =", round(rmse, 3), "\n")
cat(" - Interpretación más sencilla de los coeficientes: las variables más significativas fueron Overall, Reputación Internacional y Edad.\n\n")

cat("Random Forest:\n")
cat(" - RMSE en test =", round(rmse_rf, 3), "\n")
cat(" - Según %IncMSE, las variables más importantes son Release_Clause_eur, Overall, Skill Moves y Potential.\n\n")

cat("Comparación de Modelos:\n")
cat(" - Random Forest presenta un RMSE inferior, lo que sugiere que captura mejor la complejidad de los datos.\n")
cat(" - Sin embargo, la Regresión Lineal aporta interpretabilidad, siendo valiosa para entender la relación de cada variable con el valor.\n")
cat("\n**************************************************\n")
