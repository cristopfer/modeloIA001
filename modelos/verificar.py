import os

print("🔍 Verificando ubicación del modelo...")
modelo_dir = os.path.dirname(os.path.abspath(__file__))
ruta_modelo = os.path.join(modelo_dir, 'best_attention_unet_model.h5')

print(f"📁 Directorio actual: {modelo_dir}")
print(f"📄 Ruta del modelo: {ruta_modelo}")
print(f"✅ ¿Existe el archivo? {os.path.exists(ruta_modelo)}")

# Verificar tamaño del archivo
if os.path.exists(ruta_modelo):
    tamaño = os.path.getsize(ruta_modelo) / (1024 * 1024)  # MB
    print(f"📊 Tamaño del archivo: {tamaño:.2f} MB")