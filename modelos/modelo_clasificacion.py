# modelo_clasificacion.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import os

class ClasificadorProstata:
    def __init__(self, model_path='checkpoint_epochdn_8.pth.tar'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Usando dispositivo: {self.device}")
        
        self.model_path = model_path
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.class_names = ['neg', 'pos']
        
        # Cargar modelo automáticamente al inicializar
        self._cargar_modelo()
        
    def _cargar_modelo(self):
        """Carga el modelo entrenado"""
        try:
            # Buscar el archivo del modelo
            if not os.path.exists(self.model_path):
                # Buscar en el directorio modelo
                modelo_dir = os.path.dirname(__file__)
                ruta_alternativa = os.path.join(modelo_dir, 'checkpoint_epochdn_8.pth.tar')
                if os.path.exists(ruta_alternativa):
                    self.model_path = ruta_alternativa
                else:
                    raise FileNotFoundError(f"No se encontró el modelo en: {self.model_path}")
            
            print(f"📂 Cargando modelo desde: {self.model_path}")
            
            # Crear modelo
            self.model = timm.create_model("densenet121", pretrained=False, num_classes=2)
            
            # Cargar checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Manejar diferentes formats de checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Cargar state dict
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Modelo DenseNet121 cargado exitosamente")
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            self.model = None
    
    def predecir_imagen(self, image_path):
        """
        Realiza la predicción sobre una imagen y devuelve el resultado estructurado
        
        Args:
            image_path (str): Ruta a la imagen a analizar
            
        Returns:
            dict: Resultado estructurado con la clasificación (SIN ÁREA)
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
            
            print(f"🔍 Procesando imagen: {os.path.basename(image_path)}")
            
            # Cargar y preprocesar imagen
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Realizar predicción
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                max_prob, predicted_class = torch.max(probabilities, 1)
            
            # Obtener resultados
            clase_predicha = self.class_names[predicted_class.item()]
            confianza = max_prob.item()
            probabilidades = probabilities.cpu().numpy()[0]
            
            print(f"📊 Resultado crudo - Clase: {clase_predicha}, Confianza: {confianza:.4f}")
            
            # Generar resultado estructurado
            resultado = self._generar_resultado_estructurado(clase_predicha, confianza, probabilidades)
            
            return resultado
            
        except Exception as e:
            print(f"❌ Error procesando la imagen: {e}")
            return None
    
    def _generar_resultado_estructurado(self, clase_predicha, confianza, probabilidades):
        """
        Genera el formato de resultado estructurado para la API
        
        Args:
            clase_predicha (str): Clase predicha ('neg' o 'pos')
            confianza (float): Confianza de la predicción
            probabilidades (numpy.array): Array de probabilidades para cada clase
            
        Returns:
            dict: Resultado en formato estructurado SIN ÁREA
        """
        # Probabilidad de clase positiva (anomalía)
        prob_positiva = probabilidades[1]
        porcentaje_positivo = int(prob_positiva * 100)
        
        # Determinar nivel de riesgo basado en la probabilidad
        if prob_positiva >= 0.8:
            riesgo = "Alto Riesgo"
            riesgo_texto = "Se encontraron hallazgos significativos que requieren evaluación médica prioritaria"
            clasificacion = "PIRADS 5"
            recomendacion = "Consulte con un urólogo en las próximas 2 semanas"
        elif prob_positiva >= 0.6:
            riesgo = "Riesgo Moderado"
            riesgo_texto = "Se encontraron signos que requieren una evaluación médica en los próximos meses"
            clasificacion = "PIRADS 4"
            recomendacion = "Programe una cita con un urólogo de su preferencia"
        elif prob_positiva >= 0.4:
            riesgo = "Riesgo Bajo"
            riesgo_texto = "Hallazgos menores que pueden ser evaluados en consulta de rutina"
            clasificacion = "PIRADS 3"
            recomendacion = "Siga las recomendaciones de su médico tratante"
        else:
            riesgo = "Riesgo Muy Bajo"
            riesgo_texto = "No se encontraron hallazgos significativos"
            clasificacion = "PIRADS 1-2"
            recomendacion = "Continuar con seguimiento según indicaciones médicas"
        
        # ✅ MODIFICADO: ELIMINADO el campo "area" - ahora lo proporciona la segmentación
        
        # Construir resultado final
        resultado = {
            "riesgo": riesgo,
            "riesgoTexto": riesgo_texto,
            # "area": ELIMINADO - lo proporciona segmentación
            "probabilidad": f"{porcentaje_positivo}%",
            "clasificacion": clasificacion,
            "recomendacion": recomendacion
        }
        
        print(f"🎯 Resultado estructurado generado (SIN ÁREA):")
        print(f"   - Clasificación: {clasificacion}")
        print(f"   - Probabilidad: {porcentaje_positivo}%")
        print(f"   - Riesgo: {riesgo}")
        # No se imprime área
        
        return resultado
    
    def get_info(self):
        """Obtiene información del modelo"""
        return {
            'nombre': 'DenseNet121',
            'dispositivo': str(self.device),
            'clases': self.class_names,
            'estado': 'cargado' if self.model is not None else 'no cargado'
        }

# Función de conveniencia para uso directo (opcional)
def clasificar_imagen_prostata(image_path, model_path=None):
    """
    Función conveniente para clasificar una imagen sin crear instancia manual
    
    Args:
        image_path (str): Ruta a la imagen
        model_path (str, optional): Ruta al modelo. Si es None, usa la ruta por defecto
        
    Returns:
        dict: Resultado de la clasificación o None si hay error
    """
    try:
        if model_path:
            clasificador = ClasificadorProstata(model_path)
        else:
            clasificador = ClasificadorProstata()
        
        return clasificador.predecir_imagen(image_path)
    except Exception as e:
        print(f"❌ Error en clasificación: {e}")
        return None

# Instancia global para reutilización
_clasificador_global = None

def get_clasificador():
    """
    Obtiene la instancia global del clasificador (patrón singleton)
    
    Returns:
        ClasificadorProstata: Instancia del clasificador
    """
    global _clasificador_global
    if _clasificador_global is None:
        _clasificador_global = ClasificadorProstata()
    return _clasificador_global

if __name__ == "__main__":
    # Prueba del clasificador
    print("🔬 Clasificador de Imágenes de Próstata - DenseNet121")
    print("=" * 50)
    
    # Verificar archivos del modelo
    archivo_modelo = 'checkpoint_epochdn_8.pth.tar'
    if os.path.exists(archivo_modelo):
        print(f"✅ {archivo_modelo} - ENCONTRADO")
    else:
        # Buscar en directorio modelo
        modelo_dir = os.path.dirname(__file__)
        ruta_modelo = os.path.join(modelo_dir, archivo_modelo)
        if os.path.exists(ruta_modelo):
            print(f"✅ {archivo_modelo} - ENCONTRADO en directorio modelo")
        else:
            print(f"❌ {archivo_modelo} - NO ENCONTRADO")
    
    # Probar inicialización
    clasificador = ClasificadorProstata()
    
    if clasificador.model is not None:
        print("✅ Modelo inicializado correctamente")
        
        # Mostrar información
        info = clasificador.get_info()
        print(f"📋 Información del modelo:")
        print(f"   - Nombre: {info['nombre']}")
        print(f"   - Dispositivo: {info['dispositivo']}")
        print(f"   - Clases: {info['clases']}")
        print(f"   - Estado: {info['estado']}")
        
    else:
        print("❌ No se pudo inicializar el modelo")
    
    print("\n📝 Uso desde app.py:")
    print("   from modelo_clasificacion import ClasificadorProstata")
    print("   clasificador = ClasificadorProstata()")
    print("   resultado = clasificador.predecir_imagen('ruta/imagen.jpg')")