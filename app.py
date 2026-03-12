from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)


def detect_document_bounds(image_base64):
    """
    Определяет границы документа и угол поворота на изображении.
    Возвращает: cropX, cropY, cropWidth, cropHeight, rotation
    """
    # Декодируем base64 в изображение
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]
    
    image_bytes = base64.b64decode(image_base64)
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Не удалось декодировать изображение")
    
    original_height, original_width = image.shape[:2]
    
    # Конвертируем в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Применяем размытие для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Детекция краев Canny
    edges = cv2.Canny(blurred, 50, 150, apertureScale=3)
    
    # Морфологические операции для закрытия разрывов
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Находим контуры
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Сортируем контуры по площади (убывание)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Ищем наибольший четырехугольный контур (документ)
    doc_contour = None
    for contour in contours:
        # Аппроксимируем контур до полигона
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Если у контура 4 угла — это документ
        if len(approx) == 4:
            doc_contour = approx
            break
    
    # Если не нашли 4-угольник, берем наибольший контур
    if doc_contour is None and len(contours) > 0:
        doc_contour = contours[0]
    
    # Если контур найден, вычисляем границы
    if doc_contour is not None:
        # Находим ограничивающий прямоугольник
        x, y, w, h = cv2.boundingRect(doc_contour)
        
        # Вычисляем угол поворота
        rotation = 0
        if len(doc_contour) >= 2:
            # Находим минимальный ограничивающий прямоугольник
            rect = cv2.minAreaRect(doc_contour)
            angle = rect[-1]
            
            # Корректируем угол (OpenCV возвращает угол от -45 до 0)
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            rotation = angle
        
        # Добавляем небольшой padding
        padding = 10
        crop_x = max(0, x - padding)
        crop_y = max(0, y - padding)
        crop_width = min(w + padding * 2, original_width - crop_x)
        crop_height = min(h + padding * 2, original_height - crop_y)
        
        return {
            'cropX': int(crop_x),
            'cropY': int(crop_y),
            'cropWidth': int(crop_width),
            'cropHeight': int(crop_height),
            'rotation': round(rotation, 2),
            'originalWidth': int(original_width),
            'originalHeight': int(original_height)
        }
    
    # Если контур не найден — возвращаем полное изображение
    return {
        'cropX': 0,
        'cropY': 0,
        'cropWidth': int(original_width),
        'cropHeight': int(original_height),
        'rotation': 0,
        'originalWidth': int(original_width),
        'originalHeight': int(original_height)
    }


def detect_with_projection(image_base64):
    """
    Более точный метод с перспективной трансформацией.
    """
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]
    
    image_bytes = base64.b64decode(image_base64)
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Не удалось декодировать изображение")
    
    original_height, original_width = image.shape[:2]
    
    # Конвертируем в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Размытие
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Детекция краев
    edges = cv2.Canny(blurred, 75, 200)
    
    # Морфологические операции
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Находим контуры
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    doc_contour = None
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            doc_contour = approx
            break
    
    if doc_contour is not None:
        # Сортируем точки: верхний-левый, верхний-правый, нижний-правый, нижний-левый
        pts = doc_contour.reshape(4, 2)
        
        # Находим ограничивающий прямоугольник
        x, y, w, h = cv2.boundingRect(doc_contour)
        
        # Вычисляем угол поворота по первой стороне
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        rotation = np.degrees(np.arctan2(dy, dx))
        
        padding = 10
        crop_x = max(0, x - padding)
        crop_y = max(0, y - padding)
        crop_width = min(w + padding * 2, original_width - crop_x)
        crop_height = min(h + padding * 2, original_height - crop_y)
        
        return {
            'cropX': int(crop_x),
            'cropY': int(crop_y),
            'cropWidth': int(crop_width),
            'cropHeight': int(crop_height),
            'rotation': round(rotation, 2),
            'originalWidth': int(original_width),
            'originalHeight': int(original_height),
            'corners': pts.tolist()
        }
    
    return {
        'cropX': 0,
        'cropY': 0,
        'cropWidth': int(original_width),
        'cropHeight': int(original_height),
        'rotation': 0,
        'originalWidth': int(original_width),
        'originalHeight': int(original_height)
    }


@app.route('/detect', methods=['POST'])
def detect():
    """
    API эндпоинт для детекции границ документа.
    Принимает: {"imageBase64": "data:image/jpeg;base64,..."}
    Возвращает: {"cropX": x, "cropY": y, "cropWidth": w, "cropHeight": h, "rotation": angle}
    """
    try:
        data = request.get_json()
        
        if not data or 'imageBase64' not in data:
            return jsonify({'error': 'imageBase64 не передан'}), 400
        
        image_base64 = data['imageBase64']
        
        # Детекция границ
        result = detect_with_projection(image_base64)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Проверка работоспособности сервиса"""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("🚀 Запуск сервиса детекции границ документа...")
    print(" URL: http://localhost:5000")
    print("📡 Эндпоинт: POST /detect")
    app.run(host='0.0.0.0', port=5000, debug=False)
