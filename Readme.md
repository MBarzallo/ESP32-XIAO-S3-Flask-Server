# ESP32-XIAO-S3 / ESP32-CAM – Flask Video Processing Server

Este repositorio contiene una aplicación web desarrollada con **Python + Flask** para procesar en tiempo real el video capturado por un módulo **ESP32-CAM** o **ESP32-XIAO-S3**.  
La plataforma implementa técnicas fundamentales de **visión por computador**, incluyendo:

- Sustracción de fondo por mediana  
- Generación de ruido Gaussiano y Speckle  
- Filtros de reducción de ruido (OpenCV + PyTorch)  
- Detección de bordes (Canny y Sobel)  
- Operaciones morfológicas aplicadas a imágenes médicas  

---

## ✨ Características Principales

### ✔️ 1. Streaming en Tiempo Real
La aplicación recibe un stream MJPEG vía HTTP y muestra un panel compuesto por:

- Imagen original  
- Fondo estimado  
- Máscara binaria  
- HistEq  
- CLAHE  
- Filtro Bilateral  
- Resultado del foreground  

Todo en una cuadrícula organizada 3×3.

---

### ✔️ 2. Sustracción de Fondo (Background Subtraction)

El fondo se calcula mediante la **mediana de 40 fotogramas**, usando un buffer FIFO (`deque`).

Flujo aplicado:
1. Acumular frames
2. Calcular mediana
3. Aplicar desenfoque Gaussiano
4. Obtener diferencia absoluta
5. Umbral adaptativo basado en media
6. Apertura + Cierre + Dilatación
7. Aplicación de la máscara al frame original

---

### ✔️ 3. Simulación de Ruido + Filtros

El sistema permite simular ruido agregando:

- Ruido **Gaussiano**  
- Ruido **Speckle**  

Parámetros ajustables desde Flask:
- Media (mean)  
- Desviación estándar (std)  
- Varianza Speckle  

Filtros aplicados:
- Mediana 5×5  
- Gaussiano 7×7  
- Blur 7×7  
- **Filtro personalizado en PyTorch**  
- Canny  
- Sobel  

Kernel usado en PyTorch:

```python
kernel = [
    [0, -1/5, 0],
    [-1/5, 2.2, -1/5],
    [0, -1/5, 0]
]
```

---

### ✔️ 4. Operaciones Morfológicas (Imágenes Médicas)

A tres imágenes médicas se les aplican:

- Erosión  
- Dilatación  
- Top Hat  
- Black Hat  
- Mejoramiento: `img + (tophat - blackhat)`  

Se prueban **tres tamaños de kernel**:

- 15×15  
- 25×25  
- 37×37  

Los resultados se organizan en un panel de **3 filas × 5 columnas** para fácil comparación.

---

## 📁 Estructura del Proyecto

```
/static
    /medicas
    /templates
app.py
background.py
README.md
```



---

## 🧪 Rutas Disponibles

| Ruta | Descripción |
|------|-------------|
| `/` | Panel de sustracción de fondo |
| `/video_stream` | Stream procesado (filtros básicos) |
| `/ruido` | Controles de ruido y filtros |
| `/video_noise_stream` | Stream con ruido + filtros |
| `/morfologia` | Selección de imágenes médicas |
| `/morfologia_process/<imagen>` | Procesamiento morfológico |

---

## 🔧 Tecnologías Utilizadas

- Python 3.x  
- Flask  
- OpenCV  
- NumPy  
- PyTorch  
- ESP32-CAM / ESP32-XIAO-S3  

---

## 🔗 Repositorio en GitHub

https://github.com/MBarzallo/ESP32-XIAO-S3-Flask-Server

---

## 🧑‍🏫 Información Académica

- **Materia:** Visión por Computador  
- **Universidad:** Universidad Politécnica Salesiana (UPS) – Sede Cuenca  
- **Docente:** Ing. Vladimir Robles  
- **Autores:** Mateo Barzallo, Karen Quito  

---
