from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import math

app = Flask(__name__)


def detect_angle_hough(image):
    """
    Определяет угол поворота по линиям текста (Hough Line Transform).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Детекция краев
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough Lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                            minLineLength=80, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Пропускаем вертикальные и слишком короткие
            dx = x2 - x1
            dy = y2 - y1
            
            if abs(dx) < 20:
                continue
            
            angle = math.degrees(math.atan2(dy, dx))
            
            # Нормализуем к горизонтальным линиям
            if abs(angle) > 45:
                angle = angle - 90 if angle > 0 else angle + 90
            
            # Берём только небольшие углы (документ не перевёрнут на 90°)
            if abs(angle) < 25:
                angles.append(angle)
    
    # Средний угол
    if len(angles) > 0:
        return sum(angles) / len(angles)
    
    return 0


def detect_text_bounds(image):
    """
    Находит границы текста на изображении.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Порог для тёмного текста
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Морфология
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=3)
    eroded = cv2.erode(dilated, kernel, iterations=2)
    
    # Контуры
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Собираем все точки текста
    all_points = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300:  # Минимальная площадь
            for point in contour:
                all_points.append(point[0])
    
    if len(all_points) > 0:
        all_points = np.array(all_points)
        x, y, w, h = cv2.boundingRect(all_points)
        return x, y, w, h
    
    # Если текст не найден — всё изображение
    height, width = image.shape[:2]
    return 0, 0, width, height


@app.route('/detect', methods=['POST'])
def detect():
    """
    API для детекции угла и границ документа.
    """
    try:
        data = request.get_json()
        
        if not data or 'imageBase64' not in data:
            return jsonify({'error': 'imageBase64 не передан'}), 400
        
        image_base64 = data['imageBase64']
        
        # Декодируем
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_bytes = base64.b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Не удалось декодировать изображение'}), 400
        
        original_height, original_width = image.shape[:2]
        
        # 1. Определяем угол по линиям текста
        rotation = detect_angle_hough(image)
        
        # 2. Находим границы текста
        x, y, w, h = detect_text_bounds(image)
        
        # 3. Добавляем padding
        padding = 30
        crop_x = max(0, x - padding)
        crop_y = max(0, y - padding)
        crop_w = min(w + padding * 2, original_width - crop_x)
        crop_h = min(h + padding * 2, original_height - crop_y)
        
        return jsonify({
            'rotation': round(rotation, 2),
            'cropX': int(crop_x),
            'cropY': int(crop_y),
            'cropWidth': int(crop_w),
            'cropHeight': int(crop_h),
            'originalWidth': int(original_width),
            'originalHeight': int(original_height),
            'debug': f'Угол: {round(rotation, 2)}°, Границы: {crop_x},{crop_y},{crop_w},{crop_h}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'trace': str(type(e))}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("🚀 Запуск сервиса детекции документов...")
    print(" URL: http://localhost:5000")
    print(" Эндпоинт: POST /detect")
    app.run(host='0.0.0.0', port=5000, debug=False)
