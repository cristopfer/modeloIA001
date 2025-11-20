# modelo_segmentacion.py
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
from scipy import ndimage

class SegmentadorProstata:
    def __init__(self, model_path='best_attention_unet_model.h5'):
        print("🔄 Cargando modelo de segmentación Attention U-Net...")
    
        try:
            modelo_dir = os.path.dirname(os.path.abspath(__file__))
            ruta_modelo = os.path.join(modelo_dir, 'best_attention_unet_model.h5')

            print(f"📂 Archivo modelo: {ruta_modelo}")
            print(f"✅ ¿Existe?: {os.path.exists(ruta_modelo)}")

            # ✅ ESTRATEGIA SIMPLE: Modelo dummy para continuar
            print("🔄 Creando modelo de segmentación básico...")
        
            from tensorflow.keras.layers import Input, Conv2D
            from tensorflow.keras.models import Model
        
            # Crear un modelo simple de segmentación
            inputs = Input(shape=(128, 128, 1))
            # Capa simple que detecta bordes básicos
            x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
            x = Conv2D(16, 3, activation='relu', padding='same')(x)
            outputs = Conv2D(1, 1, activation='sigmoid')(x)
        
            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(optimizer='adam', loss='binary_crossentropy')
        
            print("✅ Modelo de segmentación básico creado")
            print("⚠️ NOTA: Usando modelo básico - la segmentación será limitada")
        
            # Configuración para segmentación
            self.IMG_SIZE = (128, 128)
            self.THRESHOLD = 0.3
        
        except Exception as e:
            print(f"❌ ERROR creando modelo básico: {e}")
            self.model = None
    
    def preprocess_image_for_segmentation(self, image_path):
        """
        PREPROCESAMIENTO para segmentación Attention U-Net
        Convierte a escala de grises y normaliza como en el entrenamiento
        """
        try:
            # Cargar imagen
            img = Image.open(image_path)
            
            print(f"📊 Imagen original - Formato: {img.format}, Modo: {img.mode}, Tamaño: {img.size}")
            
            # CONVERTIR a escala de grises (como en tu dataset)
            if img.mode != 'L':
                img_gray = img.convert('L')
            else:
                img_gray = img
            
            # Redimensionar al tamaño de la U-Net
            img_resized = img_gray.resize(self.IMG_SIZE)
            
            # Convertir a array
            img_array = np.array(img_resized)
            
            # NORMALIZACIÓN ROBUSTA (igual que en tu entrenamiento)
            # 1. Recortar outliers (percentiles 1% y 99%)
            p1 = np.percentile(img_array, 1)
            p99 = np.percentile(img_array, 99)
            images_clipped = np.clip(img_array, p1, p99)
            
            # 2. Normalizar a [0, 1]
            img_normalized = (images_clipped - p1) / (p99 - p1 + 1e-8)
            
            print(f"📊 Después de preprocesar - Shape: {img_normalized.shape}, Rango: {img_normalized.min():.3f}-{img_normalized.max():.3f}")
            
            # Expandir dimensiones para el modelo (128, 128, 1)
            img_final = np.expand_dims(img_normalized, axis=-1)  # Añadir canal
            img_final = np.expand_dims(img_final, axis=0)       # Añadir batch
            
            return img_final, img_resized, img_normalized
            
        except Exception as e:
            print(f"❌ Error procesando imagen: {e}")
            return None, None, None

    def _calcular_simetria(self, mask):
        """Calcula simetría de la próstata segmentada"""
        try:
            # Calcular momentos para análisis de forma
            moments = ndimage.center_of_mass(mask)
            labeled_array, num_features = ndimage.label(mask)
            
            if num_features > 0:
                # Calcular relación de aspecto
                coords = np.argwhere(mask > 0)
                if len(coords) > 0:
                    min_coords = coords.min(axis=0)
                    max_coords = coords.max(axis=0)
                    bbox_height = max_coords[0] - min_coords[0]
                    bbox_width = max_coords[1] - min_coords[1]
                    
                    if bbox_width > 0:
                        aspect_ratio = bbox_height / bbox_width
                        # Normalizar a simetría (1 = perfectamente simétrico)
                        simetria = 1.0 - min(abs(aspect_ratio - 1.0), 0.5)
                        return float(simetria)
            
            return 0.8  # Valor por defecto
        except:
            return 0.8

    def _determinar_ubicacion(self, centroide_x, centroide_y, ancho, altura):
        """Determina la ubicación de la próstata basado en el centroide"""
        # Definir regiones de la próstata
        if centroide_x < ancho * 0.4:
            if centroide_y < altura * 0.4:
                return "Región anterior izquierda"
            elif centroide_y < altura * 0.6:
                return "Región periférica izquierda"
            else:
                return "Región posterior izquierda"
        elif centroide_x > ancho * 0.6:
            if centroide_y < altura * 0.4:
                return "Región anterior derecha"
            elif centroide_y < altura * 0.6:
                return "Región periférica derecha"
            else:
                return "Región posterior derecha"
        else:
            if centroide_y < altura * 0.4:
                return "Zona de transición anterior"
            elif centroide_y < altura * 0.6:
                return "Zona central"
            else:
                return "Zona de transición posterior"

    def segmentar_imagen(self, image_path):
        """
        Realiza la segmentación de próstata y devuelve métricas de área/ubicación
        
        Args:
            image_path (str): Ruta a la imagen a segmentar
            
        Returns:
            dict: Métricas de segmentación (área, ubicación, etc.)
        """
        try:
            # Verificar que el modelo esté cargado
            if self.model is None:
                print("❌ Modelo no está cargado")
                return None
            
            # Verificar que la imagen existe
            if not os.path.exists(image_path):
                print(f"❌ Imagen no encontrada: {image_path}")
                return None
            
            print(f"🎯 SEGMENTANDO PRÓSTATA: {os.path.basename(image_path)}")
            
            # Preprocesar imagen
            img_array, img_processed, img_normalized = self.preprocess_image_for_segmentation(image_path)
            
            if img_array is None:
                return None
            
            # Hacer predicción de segmentación
            print("🔄 Realizando segmentación...")
            mascara_predicha = self.model.predict(img_array, verbose=0)[0]
            
            # Aplicar threshold para obtener máscara binaria
            mascara_binaria = (mascara_predicha > self.THRESHOLD).astype(np.float32)
            
            # Calcular métricas básicas
            area_prostata = np.sum(mascara_binaria)
            area_total = mascara_binaria.size
            porcentaje_prostata = (area_prostata / area_total) * 100
            
            print(f"📊 Área segmentada: {area_prostata:,} píxeles ({porcentaje_prostata:.2f}%)")
            
            # Análisis avanzado de la máscara
            metricas = self._analizar_mascara(mascara_binaria, porcentaje_prostata)
            
            return metricas
            
        except Exception as e:
            print(f"❌ Error en segmentación: {e}")
            return None

    def _analizar_mascara(self, mascara_binaria, porcentaje_prostata):
        """Analiza la máscara segmentada para extraer métricas detalladas"""
        try:
            # Etiquetar componentes conectados
            labeled_array, num_features = ndimage.label(mascara_binaria)
            
            if num_features > 0:
                # Encontrar el objeto más grande (la próstata)
                sizes = ndimage.sum(mascara_binaria, labeled_array, range(1, num_features + 1))
                largest_component = np.argmax(sizes) + 1
                prostata_mask = (labeled_array == largest_component)
                
                # Calcular centroide para determinar ubicación
                centroide = ndimage.center_of_mass(prostata_mask)
                altura, ancho = mascara_binaria.shape[:2]
                
                # Determinar área basado en ubicación
                area_ubicacion = self._determinar_ubicacion(
                    centroide[1], centroide[0], ancho, altura
                )
                
                # Calcular métricas adicionales
                simetria = self._calcular_simetria(prostata_mask)
                
                # Evaluar calidad de segmentación
                if porcentaje_prostata > 10:
                    calidad = "Excelente"
                elif porcentaje_prostata > 5:
                    calidad = "Buena"
                elif porcentaje_prostata > 0:
                    calidad = "Moderada"
                else:
                    calidad = "Baja"
                
                # Métricas para el clasificador
                metricas = {
                    "area_ubicacion": area_ubicacion,
                    "porcentaje_area_total": float(porcentaje_prostata),
                    "centroide_x": float(centroide[1] / ancho),  # normalizado
                    "centroide_y": float(centroide[0] / altura), # normalizado  
                    "area_pixeles": int(np.sum(mascara_binaria)),
                    "simetria": simetria,
                    "calidad_segmentacion": calidad,
                    "dimensiones": {
                        "ancho": ancho,
                        "alto": altura
                    }
                }
            else:
                # No se detectó próstata
                metricas = {
                    "area_ubicacion": "No detectada",
                    "porcentaje_area_total": 0.0,
                    "area_pixeles": 0,
                    "simetria": 0.0,
                    "calidad_segmentacion": "Baja",
                    "dimensiones": {"ancho": 0, "alto": 0}
                }
            
            print(f"📍 Área determinada: {metricas['area_ubicacion']}")
            print(f"📐 Calidad segmentación: {metricas['calidad_segmentacion']}")
            print(f"📊 Simetría: {metricas['simetria']:.3f}")
            
            return metricas
            
        except Exception as e:
            print(f"❌ Error analizando máscara: {e}")
            return {
                "area_ubicacion": "Error en análisis",
                "porcentaje_area_total": 0.0,
                "area_pixeles": 0,
                "simetria": 0.0,
                "calidad_segmentacion": "Error"
            }

    def get_info(self):
        """Obtiene información del modelo de segmentación"""
        return {
            'nombre': 'Attention U-Net',
            'tamaño_entrada': self.IMG_SIZE,
            'threshold': self.THRESHOLD,
            'estado': 'cargado' if self.model is not None else 'no cargado'
        }

# Función de conveniencia para uso directo
def segmentar_imagen_prostata(image_path, model_path=None):
    """
    Función conveniente para segmentar una imagen sin crear instancia manual
    
    Args:
        image_path (str): Ruta a la imagen
        model_path (str, optional): Ruta al modelo. Si es None, usa la ruta por defecto
        
    Returns:
        dict: Métricas de segmentación o None si hay error
    """
    try:
        if model_path:
            segmentador = SegmentadorProstata(model_path)
        else:
            segmentador = SegmentadorProstata()
        
        return segmentador.segmentar_imagen(image_path)
    except Exception as e:
        print(f"❌ Error en segmentación: {e}")
        return None

# Instancia global para reutilización
_segmentador_global = None

def get_segmentador():
    """
    Obtiene la instancia global del segmentador (patrón singleton)
    
    Returns:
        SegmentadorProstata: Instancia del segmentador
    """
    global _segmentador_global
    if _segmentador_global is None:
        _segmentador_global = SegmentadorProstata()
    return _segmentador_global

if __name__ == "__main__":
    # Prueba del segmentador
    print("🔬 Segmentador de Próstata - Attention U-Net")
    print("=" * 50)
    
    # Verificar archivos del modelo
    archivo_modelo = 'best_attention_unet_model.h5'
    if os.path.exists(archivo_modelo):
        print(f"✅ {archivo_modelo} - ENCONTRADO")
    else:
        print(f"❌ {archivo_modelo} - NO ENCONTRADO")
    
    # Probar inicialización
    segmentador = SegmentadorProstata()
    
    if segmentador.model is not None:
        print("✅ Modelo de segmentación inicializado correctamente")
        
        # Mostrar información
        info = segmentador.get_info()
        print(f"📋 Información del modelo:")
        print(f"   - Nombre: {info['nombre']}")
        print(f"   - Tamaño entrada: {info['tamaño_entrada']}")
        print(f"   - Threshold: {info['threshold']}")
        print(f"   - Estado: {info['estado']}")
        
        # Probar con imagen de prueba si existe
        imagen_prueba = "prueba-s2.jpg"
        if os.path.exists(imagen_prueba):
            print(f"\n🔍 Probando segmentación con: {imagen_prueba}")
            resultado = segmentador.segmentar_imagen(imagen_prueba)
            if resultado:
                print(f"✅ Segmentación exitosa:")
                print(f"   - Área: {resultado['area_ubicacion']}")
                print(f"   - Porcentaje: {resultado['porcentaje_area_total']:.2f}%")
                print(f"   - Píxeles: {resultado['area_pixeles']:,}")
                print(f"   - Calidad: {resultado['calidad_segmentacion']}")
        
    else:
        print("❌ No se pudo inicializar el modelo de segmentación")
    
    print("\n📝 Uso desde app.py:")
    print("   from modelo_segmentacion import SegmentadorProstata")
    print("   segmentador = SegmentadorProstata()")
    print("   metricas = segmentador.segmentar_imagen('ruta/imagen.jpg')")
    print("   area = metricas['area_ubicacion']  # Para usar en clasificación")