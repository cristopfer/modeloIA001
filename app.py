import os
from datetime import datetime
from flask_cors import CORS  
import gc

# Crear carpeta para uploads si no existe
os.makedirs('temp_uploads', exist_ok=True)

from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modelos.modelo_clasificacion import ClasificadorProstata
from modelos.modelo_segmentacion import SegmentadorProstata  # ✅ NUEVO: Importar segmentador

# ✅ MODIFICADO: Ambos modelos serán None hasta que se usen
clasificador = None
segmentador = None  # ✅ NUEVO: Segmentador

def get_clasificador():
    """Carga el modelo de clasificación SOLO cuando se necesita por primera vez"""
    global clasificador
    if clasificador is None:
        print("🔄 Cargando modelo de CLASIFICACIÓN bajo demanda...")
        try:
            clasificador = ClasificadorProstata()
            gc.collect()  # Liberar memoria
            print("✅ Modelo de CLASIFICACIÓN cargado correctamente")
        except Exception as e:
            print(f"❌ Error cargando modelo de clasificación: {e}")
            clasificador = None
    return clasificador

def get_segmentador():
    """Carga el modelo de segmentación SOLO cuando se necesita por primera vez"""
    global segmentador
    if segmentador is None:
        print("🔄 Cargando modelo de SEGMENTACIÓN bajo demanda...")
        try:
            segmentador = SegmentadorProstata()
            
            # ✅ VERIFICACIÓN ESTRICTA CORREGIDA
            if segmentador is None:
                print("❌ FALLO: Segmentador es None")
                return None
            elif segmentador.model is None:
                print("❌ FALLO: Modelo de segmentación es None")
                segmentador = None
                return None
            else:
                print("✅ Modelo de SEGMENTACIÓN cargado y verificado correctamente")
                gc.collect()
                
        except Exception as e:
            print(f"❌ ERROR IRRECUPERABLE en modelo de segmentación: {e}")
            segmentador = None
            return None
            
    return segmentador

def unload_modelos():
    """Descarga los modelos para liberar memoria"""
    global clasificador, segmentador
    if clasificador is not None:
        print("🗑️ Descargando modelo de CLASIFICACIÓN para liberar memoria...")
        clasificador = None
    if segmentador is not None:
        print("🗑️ Descargando modelo de SEGMENTACIÓN para liberar memoria...")
        segmentador = None
    gc.collect() 

app = Flask(__name__)

CORS(app, supports_credentials=True)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'tu-clave-secreta-aqui')
app.config['UPLOAD_FOLDER'] = 'temp_uploads'

# ✅ INICIALIZACIÓN ÚNICA de base de datos
print("🚀 Inicializando aplicación...")
try:
    # Inicializar el pool de conexiones UNA SOLA VEZ
    init_db(app)
    print("✅ Base de datos inicializada correctamente")
except Exception as e:
    print(f"❌ Error inicializando BD: {e}")

# Headers CORS manuales
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Origin,X-Requested-With,ngrok-skip-browser-warning')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Max-Age', '86400')
    return response

@app.route('/', methods=['OPTIONS'])
@app.route('/api/<path:path>', methods=['OPTIONS'])
def options_response(path=None):
    return jsonify({"status": "OK"}), 200

@app.route('/')
def health_check():
    """Endpoint de salud de la API"""
    db_status = "connected" if test_connection() else "disconnected"
    
    ia_status = "not_loaded"  # Siempre "not_loaded" porque se carga bajo demanda
    
    return jsonify({
        "status": "OK", 
        "message": "API funcionando",
        "database": db_status,
        "ai_model": ia_status,
        "service": "Prostate AI Backend"
    })

@app.route('/api/analizar', methods=['POST', 'OPTIONS'])
def analizar_imagen():
    """Endpoint para analizar imágenes de próstata con SEGMENTACIÓN + CLASIFICACIÓN - CARGA BAJO DEMANDA"""
    print("✅ /api/analizar ENDPOINT ACCEDIDO")
    if request.method == 'OPTIONS':
        return jsonify({"status": "OK"}), 200
        
    try:
        print("🚀 Solicitando análisis - cargando modelos bajo demanda...")
        
        # 1. Cargar modelo de CLASIFICACIÓN (siempre necesario)
        clasificador_local = get_clasificador()
        if clasificador_local is None or clasificador_local.model is None:
            return jsonify({
                "success": False,
                "error": "Modelo de CLASIFICACIÓN no disponible"
            }), 503

        # 2. Intentar cargar SEGMENTACIÓN (opcional - si falla, continuamos)
        segmentador_local = None
        segmentacion_disponible = False
        
        try:
            segmentador_local = get_segmentador()
            if segmentador_local and segmentador_local.model is not None:
                segmentacion_disponible = True
                print("✅ Segmentación disponible")
            else:
                print("⚠️ Segmentación no disponible - continuando sin ella")
        except Exception as seg_error:
            print(f"⚠️ Error cargando segmentación: {seg_error} - continuando sin ella")

        # Verificar si se envió un archivo
        if 'imagen' not in request.files:
            return jsonify({
                "success": False,
                "error": "No se envió ninguna imagen. Use el campo 'imagen'"
            }), 400

        file = request.files['imagen']
        
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No se seleccionó ningún archivo"
            }), 400

        # Verificar extensión del archivo
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prostata_{timestamp}_{file.filename}"
            
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)
            print(f"📁 Imagen guardada temporalmente: {filename}")
            
            try:
                print("🔬 Iniciando análisis...")
                
                # ✅ FLUJO ROBUSTO: CLASIFICACIÓN (siempre) + SEGMENTACIÓN (si disponible)
                
                # 1. PRIMERO: Clasificación para obtener diagnóstico (SIEMPRE)
                print("🔍 FASE 1: Clasificando hallazgos...")
                resultado_clasificacion = clasificador_local.predecir_imagen(filepath)
                
                if not resultado_clasificacion:
                    return jsonify({
                        "success": False,
                        "error": "El modelo no pudo clasificar la imagen"
                    }), 500
                
                # 2. LUEGO: Segmentación para obtener área real (SI ESTÁ DISPONIBLE)
                area_real = resultado_clasificacion.get("area", "Área no disponible")
                metricas_segmentacion = {"nota": "Segmentación no disponible"}
                
                if segmentacion_disponible and segmentador_local and segmentador_local.model is not None:
                    try:
                        print("🎯 FASE 2: Segmentando próstata...")
                        metricas_seg = segmentador_local.segmentar_imagen(filepath)
                        
                        if metricas_seg:
                            area_real = metricas_seg.get("area_ubicacion", "Área no determinada")
                            metricas_segmentacion = {
                                "porcentajeArea": metricas_seg.get("porcentaje_area_total", 0),
                                "areaPixeles": metricas_seg.get("area_pixeles", 0),
                                "simetria": metricas_seg.get("simetria", 0),
                                "calidad": metricas_seg.get("calidad_segmentacion", "No disponible")
                            }
                            print(f"📍 Área segmentada: {area_real}")
                        else:
                            area_real = "Segmentación falló"
                            metricas_segmentacion = {"error": "Segmentación no produjo resultados"}
                            print("⚠️ Segmentación no produjo resultados")
                            
                    except Exception as seg_error:
                        print(f"⚠️ Error durante segmentación: {seg_error}")
                        area_real = resultado_clasificacion.get("area", "Área no disponible")
                        metricas_segmentacion = {"error": f"Error en segmentación: {str(seg_error)}"}
                else:
                    print("ℹ️  Usando área de clasificación (segmentación no disponible)")
                    metricas_segmentacion = {"nota": "Segmentación no disponible - usando área de clasificación"}
                
                # 3. COMBINAR RESULTADOS
                resultado_final = {
                    "riesgo": resultado_clasificacion["riesgo"],
                    "riesgoTexto": resultado_clasificacion["riesgoTexto"],
                    "area": area_real,  # De segmentación (si disponible) o de clasificación
                    "probabilidad": resultado_clasificacion["probabilidad"],
                    "clasificacion": resultado_clasificacion["clasificacion"],
                    "recomendacion": resultado_clasificacion["recomendacion"],
                    "metricasSegmentacion": metricas_segmentacion
                }
                
                # Mensaje informativo basado en disponibilidad de segmentación
                if segmentacion_disponible and metricas_segmentacion.get("error") is None and "nota" not in metricas_segmentacion:
                    mensaje = "Análisis integrado (segmentación + clasificación) completado exitosamente"
                    print(f"✅ Análisis INTEGRADO completado:")
                    print(f"   - Clasificación: {resultado_final['clasificacion']}")
                    print(f"   - Área real: {resultado_final['area']}")
                    if 'porcentajeArea' in metricas_segmentacion:
                        print(f"   - Porcentaje área: {metricas_segmentacion['porcentajeArea']:.2f}%")
                else:
                    mensaje = "Análisis de clasificación completado exitosamente (segmentación no disponible)"
                    print(f"✅ Análisis de CLASIFICACIÓN completado:")
                    print(f"   - Clasificación: {resultado_final['clasificacion']}")
                    print(f"   - Probabilidad: {resultado_final['probabilidad']}")
                    print(f"   - Área: {resultado_final['area']}")
                
                return jsonify({
                    "success": True,
                    "message": mensaje,
                    "data": resultado_final,
                    "segmentacionDisponible": segmentacion_disponible and metricas_segmentacion.get("error") is None
                }), 200
                    
            except Exception as model_error:
                print(f"❌ Error en los modelos de IA: {model_error}")
                return jsonify({
                    "success": False,
                    "error": f"Error procesando la imagen: {str(model_error)}"
                }), 500
                
            finally:
                # Limpiar archivo temporal
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        print(f"🗑️ Archivo temporal eliminado: {filename}")
                except Exception as cleanup_error:
                    print(f"⚠️ Error eliminando archivo temporal: {cleanup_error}")

        else:
            return jsonify({
                "success": False,
                "error": "Formato de archivo no permitido. Use: PNG, JPG, JPEG, GIF, BMP"
            }), 400

    except Exception as e:
        print(f"❌ Error en el servidor: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Error interno del servidor: {str(e)}"
        }), 500

@app.route('/api/ai-status')
def ai_status():
    """Endpoint para verificar estado de los modelos de IA"""
    try:
        modelos_info = {}
        
        # Información del clasificador
        if clasificador is not None and clasificador.model is not None:
            info_clasificador = clasificador.get_info() if hasattr(clasificador, 'get_info') else {}
            modelos_info["clasificacion"] = {
                "status": "loaded",
                "model_info": info_clasificador
            }
        else:
            modelos_info["clasificacion"] = {
                "status": "not_loaded",
                "message": "Modelo de clasificación disponible para carga bajo demanda"
            }
        
        # Información del segmentador
        if segmentador is not None and segmentador.model is not None:
            info_segmentador = segmentador.get_info() if hasattr(segmentador, 'get_info') else {}
            modelos_info["segmentacion"] = {
                "status": "loaded", 
                "model_info": info_segmentador
            }
        else:
            modelos_info["segmentacion"] = {
                "status": "not_loaded",
                "message": "Modelo de segmentación disponible para carga bajo demanda"
            }
            
        return jsonify({
            "status": "available",
            "message": "Modelos de IA disponibles para carga bajo demanda",
            "modelos": modelos_info
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ✅ NUEVO: Endpoint para liberar memoria manualmente
@app.route('/api/liberar-memoria', methods=['POST'])
def liberar_memoria():
    """Libera la memoria de los modelos de IA"""
    try:
        unload_modelos()
        return jsonify({
            "success": True,
            "message": "Memoria de modelos liberada correctamente"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/db-status')
def db_status():
    """Endpoint para verificar estado de la base de datos"""
    try:
        if test_connection():
            return jsonify({
                "status": "connected",
                "message": "Conexión a PostgreSQL establecida correctamente"
            })
        else:
            return jsonify({
                "status": "disconnected",
                "message": "Error de conexión a PostgreSQL"
            }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Manejo de errores global
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint no encontrado"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Error interno del servidor"
    }), 500

# Cerrar conexiones al apagar la app
import atexit
@atexit.register
def shutdown():
    close_all_connections()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print("🚀 Iniciando servidor sin cargar modelo de IA...")
    print("📝 Modelo de IA: Se cargará bajo demanda cuando se use /api/analizar")
    
    #app.run(host='0.0.0.0', port=port, debug=debug)
    app.run(host='192.168.100.23', port=port, debug=False)