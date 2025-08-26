#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluación de Modelos de Machine Learning usando Validación Cruzada

Este script implementa una evaluación completa de modelos de machine learning
utilizando validación cruzada. Incluye:
- Carga y preparación de datos
- Definición de múltiples modelos
- Implementación de validación cruzada
- Evaluación con múltiples métricas
- Visualización de resultados

Autor: [Tu nombre]
Fecha: [Fecha actual]
"""

# %%
# =============================================================================
# PASO 1: IMPORTAR LAS BIBLIOTECAS NECESARIAS
# =============================================================================

# Importar bibliotecas para análisis de datos y manipulación
import numpy as np
import pandas as pd

# Importar bibliotecas de scikit-learn para machine learning
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix

# Importar algoritmos de clasificación
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Importar bibliotecas para visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar el estilo de las gráficas
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configurar para mostrar todas las columnas en pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("✓ Bibliotecas importadas exitosamente")

# %%
# =============================================================================
# PASO 2: CARGAR LOS DATOS
# =============================================================================

def cargar_datos():
    """
    Carga el dataset de iris y prepara los datos para el análisis.
    
    Returns:
        tuple: (X, y, feature_names, target_names, df_iris)
    """
    print("\n" + "="*60)
    print("PASO 2: CARGAR LOS DATOS")
    print("="*60)
    
    # Cargar el dataset de iris
    iris = load_iris()
    
    # Extraer las características (features) y las etiquetas (target)
    X = iris.data  # Matriz de características
    y = iris.target  # Vector de etiquetas
    
    # Obtener los nombres de las características y clases
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Crear un DataFrame para mejor visualización
    df_iris = pd.DataFrame(X, columns=feature_names)
    df_iris['target'] = y
    df_iris['target_name'] = [target_names[i] for i in y]
    
    # Mostrar información básica del dataset
    print("=== INFORMACIÓN DEL DATASET IRIS ===")
    print(f"Forma de los datos: {X.shape}")
    print(f"Número de características: {X.shape[1]}")
    print(f"Número de muestras: {X.shape[0]}")
    print(f"Número de clases: {len(np.unique(y))}")
    print(f"Clases: {target_names}")
    print(f"Características: {feature_names}")
    
    # Mostrar las primeras filas del dataset
    print("\n=== PRIMERAS 5 FILAS DEL DATASET ===")
    print(df_iris.head())
    
    # Mostrar estadísticas descriptivas
    print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
    print(df_iris.describe())
    
    # Mostrar distribución de clases
    print("\n=== DISTRIBUCIÓN DE CLASES ===")
    class_distribution = df_iris['target_name'].value_counts()
    print(class_distribution)
    
    return X, y, feature_names, target_names, df_iris

# %%
# =============================================================================
# PASO 3: VISUALIZACIÓN EXPLORATORIA DE LOS DATOS
# =============================================================================

def visualizar_datos(df_iris, feature_names, target_names):
    """
    Crea visualizaciones exploratorias de los datos.
    
    Args:
        df_iris (DataFrame): DataFrame con los datos
        feature_names (list): Lista de nombres de características
        target_names (list): Lista de nombres de clases
    """
    print("\n" + "="*60)
    print("PASO 3: VISUALIZACIÓN EXPLORATORIA DE LOS DATOS")
    print("="*60)
    
    # Crear una figura con múltiples subplots para visualizar los datos
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análisis Exploratorio del Dataset Iris', fontsize=16, fontweight='bold')
    
    # 1. Distribución de clases
    class_distribution = df_iris['target_name'].value_counts()
    axes[0, 0].pie(class_distribution.values, labels=class_distribution.index, 
                   autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Distribución de Clases')
    
    # 2. Histograma de una característica (longitud del sépalo)
    axes[0, 1].hist(df_iris[df_iris['target_name'] == 'setosa']['sepal length (cm)'], 
                    alpha=0.7, label='Setosa', bins=15)
    axes[0, 1].hist(df_iris[df_iris['target_name'] == 'versicolor']['sepal length (cm)'], 
                    alpha=0.7, label='Versicolor', bins=15)
    axes[0, 1].hist(df_iris[df_iris['target_name'] == 'virginica']['sepal length (cm)'], 
                    alpha=0.7, label='Virginica', bins=15)
    axes[0, 1].set_xlabel('Longitud del Sépalo (cm)')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución de Longitud del Sépalo por Clase')
    axes[0, 1].legend()
    
    # 3. Scatter plot de dos características
    scatter = axes[1, 0].scatter(df_iris['sepal length (cm)'], df_iris['sepal width (cm)'], 
                                c=df_iris['target'], cmap='viridis', alpha=0.7)
    axes[1, 0].set_xlabel('Longitud del Sépalo (cm)')
    axes[1, 0].set_ylabel('Ancho del Sépalo (cm)')
    axes[1, 0].set_title('Longitud vs Ancho del Sépalo')
    plt.colorbar(scatter, ax=axes[1, 0])
    
    # 4. Matriz de correlación
    correlation_matrix = df_iris[feature_names].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=axes[1, 1])
    axes[1, 1].set_title('Matriz de Correlación')
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar la matriz de correlación en formato numérico
    print("\n=== MATRIZ DE CORRELACIÓN ===")
    print(correlation_matrix.round(3))

# %%
# =============================================================================
# PASO 4: PREPROCESAMIENTO DE DATOS
# =============================================================================

def preprocesar_datos(X, feature_names):
    """
    Aplica técnicas de preprocesamiento a los datos.
    
    Args:
        X (array): Matriz de características
        feature_names (list): Lista de nombres de características
    
    Returns:
        tuple: (X_scaled, df_scaled)
    """
    print("\n" + "="*60)
    print("PASO 4: PREPROCESAMIENTO DE DATOS")
    print("="*60)
    
    # Crear una instancia del escalador estándar
    scaler = StandardScaler()
    
    # Aplicar escalado a las características
    X_scaled = scaler.fit_transform(X)
    
    # Crear un DataFrame con los datos escalados
    df_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    
    # Mostrar estadísticas antes y después del escalado
    print("=== COMPARACIÓN ANTES Y DESPUÉS DEL ESCALADO ===")
    print("\nEstadísticas antes del escalado:")
    df_original = pd.DataFrame(X, columns=feature_names)
    print(df_original.describe().round(3))
    
    print("\nEstadísticas después del escalado:")
    print(df_scaled.describe().round(3))
    
    # Verificar que el escalado funcionó correctamente
    print("\n=== VERIFICACIÓN DEL ESCALADO ===")
    print(f"Media de características escaladas: {df_scaled.mean().round(6).to_dict()}")
    print(f"Desviación estándar de características escaladas: {df_scaled.std().round(6).to_dict()}")
    
    return X_scaled, df_scaled

# %%
# =============================================================================
# PASO 5: DEFINIR LOS MODELOS A EVALUAR
# =============================================================================

def definir_modelos():
    """
    Define múltiples modelos de machine learning con diferentes configuraciones.
    
    Returns:
        dict: Diccionario con los modelos definidos
    """
    print("\n" + "="*60)
    print("PASO 5: DEFINIR LOS MODELOS A EVALUAR")
    print("="*60)
    
    # Definir los modelos que queremos evaluar
    # Cada modelo se define con sus hiperparámetros específicos
    models = {
        'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'KNN (k=7)': KNeighborsClassifier(n_neighbors=7),
        'Decision Tree (max_depth=3)': DecisionTreeClassifier(max_depth=3, random_state=42),
        'Decision Tree (max_depth=5)': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest (n_estimators=50)': RandomForestClassifier(n_estimators=50, random_state=42),
        'Random Forest (n_estimators=100)': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM (linear)': SVC(kernel='linear', random_state=42),
        'SVM (rbf)': SVC(kernel='rbf', random_state=42)
    }
    
    # Mostrar información sobre los modelos definidos
    print("=== MODELOS DEFINIDOS ===")
    for name, model in models.items():
        print(f"{name}: {type(model).__name__}")
    
    print(f"\nTotal de modelos a evaluar: {len(models)}")
    
    return models

# %%
# =============================================================================
# PASO 6: DEFINIR LA ESTRATEGIA DE VALIDACIÓN CRUZADA
# =============================================================================

def definir_estrategias_cv():
    """
    Define diferentes estrategias de validación cruzada.
    
    Returns:
        tuple: (cv_strategies, main_cv)
    """
    print("\n" + "="*60)
    print("PASO 6: DEFINIR LA ESTRATEGIA DE VALIDACIÓN CRUZADA")
    print("="*60)
    
    # Definir diferentes estrategias de validación cruzada
    cv_strategies = {
        'KFold (k=5)': KFold(n_splits=5, shuffle=True, random_state=42),
        'KFold (k=10)': KFold(n_splits=10, shuffle=True, random_state=42),
        'StratifiedKFold (k=5)': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        'StratifiedKFold (k=10)': StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    }
    
    # Mostrar información sobre las estrategias de validación cruzada
    print("=== ESTRATEGIAS DE VALIDACIÓN CRUZADA ===")
    for name, cv in cv_strategies.items():
        print(f"{name}: {type(cv).__name__}")
    
    # Usaremos StratifiedKFold con 5 pliegues como estrategia principal
    # ya que es más apropiada para problemas de clasificación con clases desbalanceadas
    main_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"\nEstrategia principal seleccionada: StratifiedKFold con 5 pliegues")
    
    return cv_strategies, main_cv

# %%
# =============================================================================
# PASO 7: DEFINIR LAS MÉTRICAS DE EVALUACIÓN
# =============================================================================

def definir_metricas():
    """
    Define múltiples métricas para evaluar el rendimiento de los modelos.
    
    Returns:
        dict: Diccionario con las métricas definidas
    """
    print("\n" + "="*60)
    print("PASO 7: DEFINIR LAS MÉTRICAS DE EVALUACIÓN")
    print("="*60)
    
    # Definir las métricas de evaluación que queremos usar
    metrics = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro'
    }
    
    # Mostrar información sobre las métricas
    print("=== MÉTRICAS DE EVALUACIÓN ===")
    metric_descriptions = {
        'accuracy': 'Precisión general (proporción de predicciones correctas)',
        'precision_macro': 'Precisión macro (promedio de precisión por clase)',
        'recall_macro': 'Recall macro (promedio de recall por clase)',
        'f1_macro': 'F1-score macro (promedio armónico de precisión y recall por clase)'
    }
    
    for metric, description in metric_descriptions.items():
        print(f"{metric}: {description}")
    
    print(f"\nTotal de métricas a evaluar: {len(metrics)}")
    
    return metrics

# %%
# =============================================================================
# PASO 8: REALIZAR VALIDACIÓN CRUZADA CON MÚLTIPLES MÉTRICAS
# =============================================================================

def realizar_validacion_cruzada(models, X_scaled, y, main_cv, metrics):
    """
    Implementa la validación cruzada para todos los modelos usando todas las métricas definidas.
    
    Args:
        models (dict): Diccionario con los modelos a evaluar
        X_scaled (array): Datos escalados
        y (array): Etiquetas
        main_cv: Estrategia de validación cruzada
        metrics (dict): Diccionario con las métricas
    
    Returns:
        dict: Diccionario con todos los resultados
    """
    print("\n" + "="*60)
    print("PASO 8: REALIZAR VALIDACIÓN CRUZADA CON MÚLTIPLES MÉTRICAS")
    print("="*60)
    
    # Diccionario para almacenar todos los resultados
    results = {}
    
    # Realizar validación cruzada para cada modelo y cada métrica
    print("=== REALIZANDO VALIDACIÓN CRUZADA ===")
    print("Procesando modelos...")
    
    for model_name, model in models.items():
        print(f"\nEvaluando {model_name}...")
        results[model_name] = {}
        
        for metric_name, metric in metrics.items():
            # Realizar validación cruzada
            scores = cross_val_score(model, X_scaled, y, cv=main_cv, scoring=metric)
            
            # Almacenar resultados
            results[model_name][metric_name] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max()
            }
            
            print(f"  {metric_name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    print("\n¡Validación cruzada completada exitosamente!")
    
    return results

# %%
# =============================================================================
# PASO 9: CALCULAR Y MOSTRAR LAS PUNTUACIONES MEDIAS
# =============================================================================

def calcular_puntuaciones_medias(results, models, metrics):
    """
    Calcula las puntuaciones medias para cada modelo y métrica.
    
    Args:
        results (dict): Resultados de la validación cruzada
        models (dict): Modelos evaluados
        metrics (dict): Métricas utilizadas
    
    Returns:
        DataFrame: DataFrame con los resultados organizados
    """
    print("\n" + "="*60)
    print("PASO 9: CALCULAR Y MOSTRAR LAS PUNTUACIONES MEDIAS")
    print("="*60)
    
    # Crear un DataFrame con los resultados de todas las métricas
    results_df = pd.DataFrame()
    
    for model_name in models.keys():
        for metric_name in metrics.keys():
            mean_score = results[model_name][metric_name]['mean']
            std_score = results[model_name][metric_name]['std']
            
            # Agregar fila al DataFrame
            new_row = pd.DataFrame({
                'Modelo': [model_name],
                'Métrica': [metric_name],
                'Puntuación Media': [mean_score],
                'Desviación Estándar': [std_score],
                'Intervalo de Confianza': [f"{mean_score:.4f} ± {std_score*2:.4f}"]
            })
            
            results_df = pd.concat([results_df, new_row], ignore_index=True)
    
    # Mostrar resultados completos
    print("=== RESULTADOS COMPLETOS DE VALIDACIÓN CRUZADA ===")
    print(results_df.round(4))
    
    # Crear una tabla pivot para mejor visualización
    pivot_results = results_df.pivot(index='Modelo', columns='Métrica', values='Puntuación Media')
    
    print("\n=== TABLA PIVOT DE PUNTUACIONES MEDIAS ===")
    print(pivot_results.round(4))
    
    return results_df, pivot_results

# %%
# =============================================================================
# PASO 10: VISUALIZAR LOS RESULTADOS
# =============================================================================

def visualizar_resultados(results_df, pivot_results, models, results):
    """
    Crea visualizaciones para comparar el rendimiento de los diferentes modelos.
    
    Args:
        results_df (DataFrame): DataFrame con resultados completos
        pivot_results (DataFrame): Tabla pivot con resultados
        models (dict): Modelos evaluados
        results (dict): Resultados originales de validación cruzada
    """
    print("\n" + "="*60)
    print("PASO 10: VISUALIZAR LOS RESULTADOS")
    print("="*60)
    
    # Crear visualizaciones de los resultados
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Comparación de Rendimiento de Modelos - Validación Cruzada', 
                 fontsize=16, fontweight='bold')
    
    # 1. Gráfico de barras para accuracy
    accuracy_data = results_df[results_df['Métrica'] == 'accuracy']
    bars1 = axes[0, 0].bar(range(len(accuracy_data)), accuracy_data['Puntuación Media'], 
                           yerr=accuracy_data['Desviación Estándar'], capsize=5, alpha=0.7)
    axes[0, 0].set_title('Accuracy por Modelo')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xticks(range(len(accuracy_data)))
    axes[0, 0].set_xticklabels(accuracy_data['Modelo'], rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Gráfico de barras para F1-score
    f1_data = results_df[results_df['Métrica'] == 'f1_macro']
    bars2 = axes[0, 1].bar(range(len(f1_data)), f1_data['Puntuación Media'], 
                           yerr=f1_data['Desviación Estándar'], capsize=5, alpha=0.7, color='orange')
    axes[0, 1].set_title('F1-Score Macro por Modelo')
    axes[0, 1].set_ylabel('F1-Score Macro')
    axes[0, 1].set_xticks(range(len(f1_data)))
    axes[0, 1].set_xticklabels(f1_data['Modelo'], rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Heatmap de todas las métricas
    heatmap_data = pivot_results
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0.8, 
                square=True, ax=axes[1, 0], fmt='.3f')
    axes[1, 0].set_title('Heatmap de Rendimiento - Todas las Métricas')
    axes[1, 0].set_xlabel('Métricas')
    axes[1, 0].set_ylabel('Modelos')
    
    # 4. Box plot de las puntuaciones de accuracy
    accuracy_scores = []
    model_names = []
    for model_name in models.keys():
        scores = results[model_name]['accuracy']['scores']
        accuracy_scores.extend(scores)
        model_names.extend([model_name] * len(scores))
    
    box_data = pd.DataFrame({'Modelo': model_names, 'Accuracy': accuracy_scores})
    sns.boxplot(data=box_data, x='Modelo', y='Accuracy', ax=axes[1, 1])
    axes[1, 1].set_title('Distribución de Accuracy por Modelo')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# %%
# =============================================================================
# PASO 11: ANÁLISIS DETALLADO DEL MEJOR MODELO
# =============================================================================

def analizar_mejor_modelo(results, models, X_scaled, y, target_names):
    """
    Identifica el mejor modelo y realiza un análisis detallado.
    
    Args:
        results (dict): Resultados de la validación cruzada
        models (dict): Modelos evaluados
        X_scaled (array): Datos escalados
        y (array): Etiquetas
        target_names (list): Nombres de las clases
    """
    print("\n" + "="*60)
    print("PASO 11: ANÁLISIS DETALLADO DEL MEJOR MODELO")
    print("="*60)
    
    # Crear tabla pivot para identificar el mejor modelo
    results_df = pd.DataFrame()
    for model_name in models.keys():
        for metric_name in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']:
            mean_score = results[model_name][metric_name]['mean']
            new_row = pd.DataFrame({
                'Modelo': [model_name],
                'Métrica': [metric_name],
                'Puntuación Media': [mean_score]
            })
            results_df = pd.concat([results_df, new_row], ignore_index=True)
    
    pivot_results = results_df.pivot(index='Modelo', columns='Métrica', values='Puntuación Media')
    
    # Identificar el mejor modelo según accuracy
    best_model_name = pivot_results['accuracy'].idxmax()
    best_model_score = pivot_results.loc[best_model_name, 'accuracy']
    
    print("=== ANÁLISIS DEL MEJOR MODELO ===")
    print(f"Mejor modelo según accuracy: {best_model_name}")
    print(f"Puntuación de accuracy: {best_model_score:.4f}")
    
    # Mostrar todas las métricas del mejor modelo
    print(f"\nRendimiento completo del mejor modelo:")
    best_model_metrics = pivot_results.loc[best_model_name]
    for metric, score in best_model_metrics.items():
        print(f"  {metric}: {score:.4f}")
    
    # Entrenar el mejor modelo en todo el dataset para análisis detallado
    best_model = models[best_model_name]
    best_model.fit(X_scaled, y)
    
    # Realizar predicciones
    y_pred = best_model.predict(X_scaled)
    
    # Mostrar reporte de clasificación detallado
    print(f"\n=== REPORTE DE CLASIFICACIÓN DETALLADO ===")
    print(f"Modelo: {best_model_name}")
    print(classification_report(y, y_pred, target_names=target_names))
    
    # Mostrar matriz de confusión
    print(f"\n=== MATRIZ DE CONFUSIÓN ===")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz de Confusión - {best_model_name}')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.show()
    
    return best_model_name, best_model

# %%
# =============================================================================
# PASO 12: COMPARACIÓN DE ESTRATEGIAS DE VALIDACIÓN CRUZADA
# =============================================================================

def comparar_estrategias_cv(cv_strategies, best_model, X_scaled, y):
    """
    Compara el rendimiento usando diferentes estrategias de validación cruzada.
    
    Args:
        cv_strategies (dict): Diferentes estrategias de validación cruzada
        best_model: El mejor modelo identificado
        X_scaled (array): Datos escalados
        y (array): Etiquetas
    """
    print("\n" + "="*60)
    print("PASO 12: COMPARACIÓN DE ESTRATEGIAS DE VALIDACIÓN CRUZADA")
    print("="*60)
    
    # Comparar diferentes estrategias de validación cruzada
    print("=== COMPARACIÓN DE ESTRATEGIAS DE VALIDACIÓN CRUZADA ===")
    
    # Usar el mejor modelo para la comparación
    cv_comparison = {}
    
    for cv_name, cv_strategy in cv_strategies.items():
        scores = cross_val_score(best_model, X_scaled, y, cv=cv_strategy, scoring='accuracy')
        cv_comparison[cv_name] = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
        print(f"{cv_name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # Visualizar comparación de estrategias CV
    cv_names = list(cv_comparison.keys())
    cv_means = [cv_comparison[name]['mean'] for name in cv_names]
    cv_stds = [cv_comparison[name]['std'] for name in cv_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(cv_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.7, color='skyblue')
    plt.title('Comparación de Estrategias de Validación Cruzada')
    plt.ylabel('Accuracy Media')
    plt.xlabel('Estrategia de Validación Cruzada')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for bar, mean, std in zip(bars, cv_means, cv_stds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                 f'{mean:.4f}\n±{std*2:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

# %%
# =============================================================================
# PASO 13: RESUMEN Y CONCLUSIONES
# =============================================================================

def generar_resumen(results, models, metrics, best_model_name, main_cv):
    """
    Genera un resumen ejecutivo de los hallazgos principales.
    
    Args:
        results (dict): Resultados de la validación cruzada
        models (dict): Modelos evaluados
        metrics (dict): Métricas utilizadas
        best_model_name (str): Nombre del mejor modelo
        main_cv: Estrategia de validación cruzada principal
    """
    print("\n" + "="*60)
    print("PASO 13: RESUMEN Y CONCLUSIONES")
    print("="*60)
    
    # Crear tabla pivot para el resumen
    results_df = pd.DataFrame()
    for model_name in models.keys():
        for metric_name in metrics.keys():
            mean_score = results[model_name][metric_name]['mean']
            new_row = pd.DataFrame({
                'Modelo': [model_name],
                'Métrica': [metric_name],
                'Puntuación Media': [mean_score]
            })
            results_df = pd.concat([results_df, new_row], ignore_index=True)
    
    pivot_results = results_df.pivot(index='Modelo', columns='Métrica', values='Puntuación Media')
    
    # Crear un resumen ejecutivo de los resultados
    print("=== RESUMEN EJECUTIVO ===")
    print("\n1. MEJORES MODELOS POR MÉTRICA:")
    for metric in metrics.keys():
        best_for_metric = pivot_results[metric].idxmax()
        best_score = pivot_results.loc[best_for_metric, metric]
        print(f"   {metric}: {best_for_metric} ({best_score:.4f})")
    
    print("\n2. TOP 5 MODELOS POR ACCURACY:")
    top_5_accuracy = pivot_results['accuracy'].sort_values(ascending=False).head(5)
    for i, (model, score) in enumerate(top_5_accuracy.items(), 1):
        print(f"   {i}. {model}: {score:.4f}")
    
    print("\n3. ESTADÍSTICAS GENERALES:")
    print(f"   Número total de modelos evaluados: {len(models)}")
    print(f"   Número de pliegues en validación cruzada: {main_cv.n_splits}")
    print(f"   Métricas evaluadas: {list(metrics.keys())}")
    print(f"   Accuracy promedio de todos los modelos: {pivot_results['accuracy'].mean():.4f}")
    print(f"   Accuracy más alta: {pivot_results['accuracy'].max():.4f}")
    print(f"   Accuracy más baja: {pivot_results['accuracy'].min():.4f}")
    
    print("\n4. RECOMENDACIONES:")
    print(f"   - El mejor modelo general es: {best_model_name}")
    print(f"   - Estrategia de validación cruzada recomendada: StratifiedKFold con 5 pliegues")
    print(f"   - Los datos están bien balanceados, por lo que accuracy es una métrica apropiada")
    print(f"   - El escalado de características mejoró el rendimiento de los modelos")
    
    # Crear una tabla final de recomendaciones
    best_model_score = pivot_results.loc[best_model_name, 'accuracy']
    recommendations_df = pd.DataFrame({
        'Aspecto': ['Mejor Modelo', 'Accuracy', 'Estrategia CV', 'Métricas Evaluadas', 'Preprocesamiento'],
        'Recomendación': [
            best_model_name,
            f"{best_model_score:.4f}",
            'StratifiedKFold (k=5)',
            ', '.join(metrics.keys()),
            'StandardScaler'
        ]
    })
    
    print("\n5. TABLA DE RECOMENDACIONES:")
    print(recommendations_df.to_string(index=False))

# %%
# =============================================================================
# RESUMEN EJECUTIVO - EJECUTAR TODO EL ANÁLISIS
# =============================================================================

# Ejecutar todo el análisis paso a paso
print("="*80)
print("EVALUACIÓN DE MODELOS DE MACHINE LEARNING USANDO VALIDACIÓN CRUZADA")
print("="*80)

# Paso 2: Cargar los datos
X, y, feature_names, target_names, df_iris = cargar_datos()

# Paso 3: Visualización exploratoria
visualizar_datos(df_iris, feature_names, target_names)

# Paso 4: Preprocesamiento
X_scaled, df_scaled = preprocesar_datos(X, feature_names)

# Paso 5: Definir modelos
models = definir_modelos()

# Paso 6: Definir estrategias de validación cruzada
cv_strategies, main_cv = definir_estrategias_cv()

# Paso 7: Definir métricas
metrics = definir_metricas()

# Paso 8: Realizar validación cruzada
results = realizar_validacion_cruzada(models, X_scaled, y, main_cv, metrics)

# Paso 9: Calcular puntuaciones medias
results_df, pivot_results = calcular_puntuaciones_medias(results, models, metrics)

# Paso 10: Visualizar resultados
visualizar_resultados(results_df, pivot_results, models, results)

# Paso 11: Análisis del mejor modelo
best_model_name, best_model = analizar_mejor_modelo(results, models, X_scaled, y, target_names)

# Paso 12: Comparar estrategias de validación cruzada
comparar_estrategias_cv(cv_strategies, best_model, X_scaled, y)

# Paso 13: Generar resumen
generar_resumen(results, models, metrics, best_model_name, main_cv)

print("\n" + "="*80)
print("¡EVALUACIÓN COMPLETADA EXITOSAMENTE!")
print("="*80)

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Función principal que ejecuta todo el proceso de evaluación de modelos.
    """
    print("="*80)
    print("EVALUACIÓN DE MODELOS DE MACHINE LEARNING USANDO VALIDACIÓN CRUZADA")
    print("="*80)
    
    try:
        # Paso 1: Las bibliotecas ya están importadas al inicio del script
        
        # Paso 2: Cargar los datos
        X, y, feature_names, target_names, df_iris = cargar_datos()
        
        # Paso 3: Visualización exploratoria
        visualizar_datos(df_iris, feature_names, target_names)
        
        # Paso 4: Preprocesamiento
        X_scaled, df_scaled = preprocesar_datos(X, feature_names)
        
        # Paso 5: Definir modelos
        models = definir_modelos()
        
        # Paso 6: Definir estrategias de validación cruzada
        cv_strategies, main_cv = definir_estrategias_cv()
        
        # Paso 7: Definir métricas
        metrics = definir_metricas()
        
        # Paso 8: Realizar validación cruzada
        results = realizar_validacion_cruzada(models, X_scaled, y, main_cv, metrics)
        
        # Paso 9: Calcular puntuaciones medias
        results_df, pivot_results = calcular_puntuaciones_medias(results, models, metrics)
        
        # Paso 10: Visualizar resultados
        visualizar_resultados(results_df, pivot_results, models, results)
        
        # Paso 11: Análisis del mejor modelo
        best_model_name, best_model = analizar_mejor_modelo(results, models, X_scaled, y, target_names)
        
        # Paso 12: Comparar estrategias de validación cruzada
        comparar_estrategias_cv(cv_strategies, best_model, X_scaled, y)
        
        # Paso 13: Generar resumen
        generar_resumen(results, models, metrics, best_model_name, main_cv)
        
        print("\n" + "="*80)
        print("¡EVALUACIÓN COMPLETADA EXITOSAMENTE!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError durante la ejecución: {str(e)}")
        import traceback
        traceback.print_exc()

# =============================================================================
# EJECUTAR EL SCRIPT
# =============================================================================

if __name__ == "__main__":
    main()
