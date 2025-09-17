

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json
import numpy as np
from datetime import datetime
import threading
import time
import requests


app = Flask(__name__)


model = None
feature_names = None
request_count = 0
predictions_log = []

def load_model_and_features():
    """Cargar modelo y características al iniciar"""
    global model, feature_names
    
    try:
        # Cargar modelo entrenado
        model = joblib.load('best_model.pkl')
        print("✅ Modelo cargado exitosamente")
        
        # Cargar nombres de características
        with open('feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        print(f"✅ {len(feature_names)} características cargadas")
        
        return True
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return False

def preprocess_wine_features(wine_data):
    """Aplicar el mismo preprocesamiento que durante el entrenamiento"""
    
    df = wine_data.copy()
    
 
    df['alcohol_category'] = pd.cut(df['alcohol'], 
                                   bins=[0, 10, 12, 15, 20], 
                                   labels=['Low', 'Medium', 'High', 'Very_High'])
    
    df['acidity_ratio'] = df['fixed acidity'] / (df['volatile acidity'] + 0.001)
    df['sulfur_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 0.001)
    df['sugar_alcohol_interaction'] = df['residual sugar'] * df['alcohol']
    
    df['ph_category'] = pd.cut(df['pH'], 
                              bins=[0, 3.0, 3.3, 3.6, 5.0], 
                              labels=['Very_Acidic', 'Acidic', 'Moderate', 'Basic'])
    

    if 'wine_type' not in df.columns:
        df['wine_type'] = 'red'
    

    categorical_features = ['wine_type', 'alcohol_category', 'ph_category']
    df_encoded = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)
    
    return df_encoded

@app.route('/', methods=['GET'])
def home():
    """Endpoint de información del servicio"""
    
    global request_count, predictions_log
    
    info = {
        "service": "Wine Quality Prediction API",
        "version": "1.0",
        "model": "Random Forest Classifier",
        "status": "running" if model is not None else "model_not_loaded",
        "features_count": len(feature_names) if feature_names else 0,
        "total_requests": request_count,
        "total_predictions": len(predictions_log),
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "predict": "/score (POST)",
            "health": "/health (GET)",
            "stats": "/stats (GET)"
        }
    }
    
    return jsonify(info)

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de health check (como Azure ML)"""
    
    status = {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "features_loaded": feature_names is not None,
        "timestamp": datetime.now().isoformat()
    }
    
    return jsonify(status)

@app.route('/score', methods=['POST'])
def predict():
    """Endpoint principal de predicción (equivalente a Azure ML endpoint)"""
    
    global request_count, predictions_log
    request_count += 1
    
    try:
        # Validar que el modelo esté cargado
        if model is None:
            return jsonify({
                "error": "Model not loaded",
                "request_id": request_count
            }), 500
        
        # Parsear datos de entrada
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "request_id": request_count,
                "expected_format": {
                    "data": [
                        {
                            "fixed acidity": 7.4,
                            "volatile acidity": 0.7,
                            # ... otros campos
                        }
                    ]
                }
            }), 400
        
        # Convertir a DataFrame
        if isinstance(data, dict) and 'data' in data:
            input_data = pd.DataFrame(data['data'])
        elif isinstance(data, list):
            input_data = pd.DataFrame(data)
        else:
            input_data = pd.DataFrame([data])
        
        # Validar características básicas requeridas
        basic_features = [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]
        
        missing_features = [f for f in basic_features if f not in input_data.columns]
        if missing_features:
            return jsonify({
                "error": f"Missing required features: {missing_features}",
                "required_features": basic_features,
                "request_id": request_count
            }), 400
        
        # Aplicar preprocesamiento
        processed_data = preprocess_wine_features(input_data)
        
        # Alinear con características de entrenamiento
        for feature in feature_names:
            if feature not in processed_data.columns:
                processed_data[feature] = 0
        
        # Seleccionar características en el orden correcto
        X = processed_data[feature_names]
        
        # Hacer predicciones
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Formatear resultados
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            result = {
                "sample_id": i + 1,
                "prediction": int(pred),
                "prediction_label": "Good Wine" if pred == 1 else "Regular Wine",
                "confidence": float(max(prob)),
                "probability_regular": float(prob[0]),
                "probability_good": float(prob[1]),
                "features_processed": len(feature_names)
            }
            results.append(result)
        
        # Log de predicción
        prediction_log = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_count,
            "samples_processed": len(results),
            "predictions": [r["prediction_label"] for r in results]
        }
        predictions_log.append(prediction_log)
        
        # Respuesta final (formato compatible con Azure ML)
        response = {
            "predictions": results,
            "model_info": {
                "name": "Random Forest Wine Quality Classifier",
                "version": "1.0",
                "framework": "scikit-learn"
            },
            "request_info": {
                "request_id": request_count,
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": "< 100ms"
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        error_response = {
            "error": f"Prediction error: {str(e)}",
            "request_id": request_count,
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(error_response), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Endpoint de estadísticas del servicio"""
    
    global request_count, predictions_log
    
    # Calcular estadísticas
    total_predictions = sum(len(log["predictions"]) for log in predictions_log)
    good_wines = sum(log["predictions"].count("Good Wine") for log in predictions_log)
    regular_wines = total_predictions - good_wines
    
    stats = {
        "service_stats": {
            "total_requests": request_count,
            "total_predictions": total_predictions,
            "predictions_breakdown": {
                "good_wines": good_wines,
                "regular_wines": regular_wines,
                "good_wine_percentage": (good_wines / total_predictions * 100) if total_predictions > 0 else 0
            }
        },
        "model_info": {
            "algorithm": "Random Forest",
            "features_count": len(feature_names) if feature_names else 0,
            "trained_accuracy": "82.0%",
            "f1_score": "0.8620"
        },
        "recent_predictions": predictions_log[-5:] if len(predictions_log) > 0 else [],
        "timestamp": datetime.now().isoformat()
    }
    
    return jsonify(stats)

def test_endpoint_locally():
    """Función para probar el endpoint localmente"""
    
    # Datos de prueba
    test_data = {
        "data": [
            {
                "fixed acidity": 7.4,
                "volatile acidity": 0.7,
                "citric acid": 0.0,
                "residual sugar": 1.9,
                "chlorides": 0.076,
                "free sulfur dioxide": 11.0,
                "total sulfur dioxide": 34.0,
                "density": 0.9978,
                "pH": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4
            },
            {
                "fixed acidity": 8.1,
                "volatile acidity": 0.28,
                "citric acid": 0.4,
                "residual sugar": 6.9,
                "chlorides": 0.05,
                "free sulfur dioxide": 30.0,
                "total sulfur dioxide": 97.0,
                "density": 0.9951,
                "pH": 3.26,
                "sulphates": 0.44,
                "alcohol": 10.1
            }
        ]
    }
    
    try:
        # Esperar a que el servidor inicie
        time.sleep(2)
        
        print("\n🧪 Probando endpoint local...")
        
        # Test 1: Health check
        response = requests.get("http://localhost:5000/health")
        print(f"Health check: {response.status_code} - {response.json()['status']}")
        
        # Test 2: Predicción
        response = requests.post(
            "http://localhost:5000/score",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Predicción exitosa:")
            for pred in result["predictions"]:
                print(f"   Muestra {pred['sample_id']}: {pred['prediction_label']} (confianza: {pred['confidence']:.3f})")
        else:
            print(f"❌ Error en predicción: {response.status_code}")
            
        # Test 3: Estadísticas
        response = requests.get("http://localhost:5000/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"📊 Estadísticas: {stats['service_stats']['total_predictions']} predicciones realizadas")
        
    except Exception as e:
        print(f"Error en prueba: {e}")

def run_tests():
    """Ejecutar pruebas en hilo separado"""
    test_endpoint_locally()

def create_postman_collection():
    """Crear colección de Postman para pruebas"""
    
    postman_collection = {
        "info": {
            "name": "Wine Quality Prediction API",
            "description": "Colección para probar el endpoint local de predicción de calidad de vinos",
            "version": "1.0.0"
        },
        "item": [
            {
                "name": "Health Check",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": "http://localhost:5000/health",
                        "protocol": "http",
                        "host": ["localhost"],
                        "port": "5000",
                        "path": ["health"]
                    }
                }
            },
            {
                "name": "Predict Wine Quality",
                "request": {
                    "method": "POST",
                    "header": [
                        {
                            "key": "Content-Type",
                            "value": "application/json"
                        }
                    ],
                    "body": {
                        "mode": "raw",
                        "raw": json.dumps({
                            "data": [
                                {
                                    "fixed acidity": 7.4,
                                    "volatile acidity": 0.7,
                                    "citric acid": 0.0,
                                    "residual sugar": 1.9,
                                    "chlorides": 0.076,
                                    "free sulfur dioxide": 11.0,
                                    "total sulfur dioxide": 34.0,
                                    "density": 0.9978,
                                    "pH": 3.51,
                                    "sulphates": 0.56,
                                    "alcohol": 9.4
                                }
                            ]
                        }, indent=2)
                    },
                    "url": {
                        "raw": "http://localhost:5000/score",
                        "protocol": "http",
                        "host": ["localhost"],
                        "port": "5000",
                        "path": ["score"]
                    }
                }
            },
            {
                "name": "Get Statistics",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": "http://localhost:5000/stats",
                        "protocol": "http",
                        "host": ["localhost"],
                        "port": "5000",
                        "path": ["stats"]
                    }
                }
            }
        ]
    }
    
    # Guardar colección de Postman
    with open("wine_quality_api_postman.json", "w") as f:
        json.dump(postman_collection, f, indent=2)
    
    print("📄 Colección de Postman creada: wine_quality_api_postman.json")

def main():
    """Función principal para iniciar el servidor"""
    
    print("🍷 Wine Quality Prediction API - Simulación Local")
    print("=" * 60)
    print("Equivalente funcional a Azure ML Online Endpoint")
    print()
    
    # Cargar modelo y características
    if not load_model_and_features():
        print("❌ No se pudo cargar el modelo. Asegúrate de tener:")
        print("   - best_model.pkl")
        print("   - feature_names.txt")
        return
    
    # Crear colección de Postman
    create_postman_collection()
    
    print("🚀 Iniciando servidor de predicciones...")
    print("📍 URL: http://localhost:5000")
    print("📋 Endpoints disponibles:")
    print("   GET  /        - Información del servicio")
    print("   GET  /health  - Health check")
    print("   POST /score   - Predicciones de calidad de vino")
    print("   GET  /stats   - Estadísticas del servicio")
    print()
    print("💡 Para probar:")
    print("   1. Usa Postman con wine_quality_api_postman.json")
    print("   2. O envía POST a http://localhost:5000/score con JSON")
    print("   3. Presiona Ctrl+C para detener")
    print()
    
    # Iniciar pruebas automáticas en hilo separado
    test_thread = threading.Thread(target=run_tests)
    test_thread.daemon = True
    test_thread.start()
    
    # Iniciar servidor Flask
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Servidor detenido")
        print(f"📊 Total de requests procesados: {request_count}")
        print(f"📈 Total de predicciones realizadas: {len(predictions_log)}")

if __name__ == "__main__":
    main()