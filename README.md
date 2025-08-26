# Evaluación de Modelos de Machine Learning usando Validación Cruzada

Este proyecto implementa una evaluación completa de modelos de machine learning utilizando validación cruzada. El script evalúa múltiples algoritmos de clasificación y proporciona análisis detallados de rendimiento.

## 📋 Descripción

El proyecto incluye:
- **Carga y preparación de datos** del dataset Iris
- **Visualización exploratoria** de datos
- **Preprocesamiento** con escalado de características
- **Evaluación de 10 modelos diferentes** con validación cruzada
- **Múltiples métricas de evaluación** (accuracy, precisión, recall, F1-score)
- **Visualizaciones comprehensivas** de resultados
- **Análisis del mejor modelo** con matriz de confusión
- **Comparación de estrategias** de validación cruzada

## 🚀 Instalación

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/davidtimana/machine-learning-avanzado.git
cd machine-learning-avanzado
```

2. **Crear entorno virtual:**
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

## 📊 Uso

### Ejecutar el script principal:
```bash
python evaluacion_modelos_validacion_cruzada.py
```

### Estructura del proyecto:
```
machine-learning-avanzado/
├── evaluacion_modelos_validacion_cruzada.py  # Script principal
├── requirements.txt                          # Dependencias
├── README.md                                 # Documentación
├── venv/                                     # Entorno virtual
└── notebooks/                                # Jupyter notebooks (si los hay)
```

## 🔧 Características

### Modelos Evaluados:
- **K-Nearest Neighbors (KNN)** con k=3, 5, 7
- **Árboles de Decisión** con profundidad máxima 3 y 5
- **Random Forest** con 50 y 100 estimadores
- **Regresión Logística**
- **Support Vector Machines (SVM)** con kernels lineal y RBF

### Métricas de Evaluación:
- **Accuracy**: Precisión general
- **Precision Macro**: Precisión promedio por clase
- **Recall Macro**: Recall promedio por clase
- **F1-Score Macro**: F1-score promedio por clase

### Estrategias de Validación Cruzada:
- **KFold** con 5 y 10 pliegues
- **StratifiedKFold** con 5 y 10 pliegues (recomendado para clasificación)

## 📈 Resultados

El script genera:
1. **Análisis exploratorio** de datos con visualizaciones
2. **Tablas comparativas** de rendimiento de modelos
3. **Gráficos de barras** para comparar modelos
4. **Heatmaps** de rendimiento
5. **Box plots** de distribución de puntuaciones
6. **Matriz de confusión** del mejor modelo
7. **Resumen ejecutivo** con recomendaciones

## 🎯 Outputs Esperados

### Visualizaciones:
- Distribución de clases
- Matriz de correlación
- Comparación de modelos por métricas
- Heatmap de rendimiento
- Matriz de confusión

### Métricas de Rendimiento:
- Puntuaciones medias por modelo y métrica
- Desviaciones estándar
- Intervalos de confianza
- Ranking de modelos

## 📝 Documentación

### Funciones Principales:

1. **`cargar_datos()`**: Carga el dataset Iris y prepara los datos
2. **`visualizar_datos()`**: Crea visualizaciones exploratorias
3. **`preprocesar_datos()`**: Aplica escalado de características
4. **`definir_modelos()`**: Define los modelos a evaluar
5. **`realizar_validacion_cruzada()`**: Ejecuta validación cruzada
6. **`visualizar_resultados()`**: Genera gráficos de comparación
7. **`analizar_mejor_modelo()`**: Análisis detallado del mejor modelo
8. **`generar_resumen()`**: Crea resumen ejecutivo

## 🔍 Análisis Técnico

### Preprocesamiento:
- **StandardScaler**: Normalización de características
- Verificación de escalado (media ≈ 0, desviación ≈ 1)

### Validación Cruzada:
- **StratifiedKFold**: Mantiene proporción de clases en cada pliegue
- **5 pliegues**: Balance entre robustez y eficiencia computacional

### Selección de Modelo:
- Evaluación por múltiples métricas
- Consideración de estabilidad (desviación estándar)
- Análisis de matriz de confusión

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👨‍💻 Autor

**David Timana**
- GitHub: [@davidtimana](https://github.com/davidtimana)

## 🙏 Agradecimientos

- Dataset Iris de scikit-learn
- Comunidad de scikit-learn por las herramientas de ML
- Comunidad de Python por las bibliotecas de visualización

## 📞 Contacto

David Timana - [@davidtimana](https://github.com/davidtimana)

Link del proyecto: [https://github.com/davidtimana/machine-learning-avanzado](https://github.com/davidtimana/machine-learning-avanzado)
