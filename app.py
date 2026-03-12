from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from PIL import Image
import io
import math

app = Flask(__name__)


def order_points(pts):
    """
    Сортирует 4 точки в порядке: верхний-левый, верхний-правый, нижний-правый, нижний-левый
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Сортировка по сумме координат (верхний-левый имеет минимальную сумму, нижний-правый - максимальную)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # верхний-левый
    rect[2] = pts[np.argmax(s)]  # нижний-правый
    
    # Сортировка по разности координат (верхний-правый имеет минимальную разность, нижний-левый - максимальную)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # верхний-правый
    rect[3] = pts[np.argmax(diff)]  # нижний-левый
    
    return rect


def calculate_angle(pts):
    """
    Вычисляет угол поворота документа по верхней грани.
    Возвращает угол в градусах (положительный = поворот по часовой стрелке).
    """
    # Верхняя грань: от верхнего-левого к верхнему-правому
    top_left = pts[0]
    top_right = pts[1]
    
    dx = top_right[0] - top_left[0]
    dy = top_right[1] - top_left[1]
    
    # Вычисляем угол в градусах
    angle = math.degrees(math.atan2(dy, dx))
    
    # Нормализуем: если угол > 45°, значит документ повёрнут на 90°
    if abs(angle) > 45:
        angle = angle - 90 if angle > 0 else angle + 90
    
    return round(angle, 2)


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
    Метод детекции угла по линиям текста (Hough Line Transform).
    Лучше работает, когда документ сливается с фоном.
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
    
    # Детекция краев Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough Line Transform для поиска линий
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=100, maxLineGap=10)
    
    # Вычисляем средний угол по всем линиям
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Пропускаем вертикальные линии
            if abs(x2 - x1) < 10:
                continue
            
            # Вычисляем угол линии
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            # Нормализуем угол (горизонтальные линии ≈ 0°)
            if abs(angle) > 45:
                angle = angle - 90 if angle > 0 else angle + 90
            
            # Пропускаем слишком большие углы
            if abs(angle) > 30:
                continue
            
            angles.append(angle)
    
    # Если нашли линии, вычисляем средний угол
    rotation = 0
    if len(angles) > 0:
        rotation = sum(angles) / len(angles)
        rotation = round(rotation, 2)
    
    # Находим границы для crop (по контурам)
    # Инвертируем (тёмный текст на светлом фоне)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Морфологические операции
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)
    
    # Находим контуры
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Объединяем все контуры для определения границ текста
    all_points = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Пропускаем шум
            for point in contour:
                all_points.append(point[0])
    
    # Вычисляем границы
    if len(all_points) > 0:
        all_points = np.array(all_points)
        x, y, w, h = cv2.boundingRect(all_points)
        
        padding = 20
        crop_x = max(0, x - padding)
        crop_y = max(0, y - padding)
        crop_width = min(w + padding * 2, original_width - crop_x)
        crop_height = min(h + padding * 2, original_height - crop_y)
    else:
        # Если текст не найден, используем всё изображение
        crop_x = 0
        crop_y = 0
        crop_width = original_width
        crop_height = original_height
    
    return {
        'cropX': int(crop_x),
        'cropY': int(crop_y),
        'cropWidth': int(crop_width),
        'cropHeight': int(crop_height),
        'rotation': rotation,
        'originalWidth': int(original_width),
        'originalHeight': int(original_height),
        'linesFound': len(angles),
        'avgAngle': rotation
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
