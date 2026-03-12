from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import math

app = Flask(__name__)


def detect_angle_hough(image):
    """
    Определяет угол поворота по линиям текста.
    Возвращает угол для поворота изображения (инвертированный).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Размытие
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Детекция краев
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Hough Lines — строгий фильтр
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=100, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            dx = x2 - x1
            dy = y2 - y1
            
            # Пропускаем короткие линии
            length = math.sqrt(dx * dx + dy * dy)
            if length < 80:
                continue
            
            # Пропускаем вертикальные линии
            if abs(dx) < 30:
                continue
            
            angle = math.degrees(math.atan2(dy, dx))
            
            # Нормализуем к горизонтальным линиям
            if abs(angle) > 45:
                angle = angle - 90 if angle > 0 else angle + 90
            
            # Берём только небольшие углы (документ не перевёрнут)
            if abs(angle) < 30:
                angles.append(angle)
    
    # Средний угол
    if len(angles) > 0:
        avg_angle = sum(angles) / len(angles)
        # ИНВЕРТИРУЕМ угол для поворота изображения
        return -avg_angle, len(angles)
    
    return 0, 0


@app.route('/detect-angle', methods=['POST'])
def detect_angle():
    """
    API только для определения угла поворота.
    Возвращает: rotation (угол в градусах)
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
        
        # Определяем угол
        rotation, lines_count = detect_angle_hough(image)
        
        return jsonify({
            'rotation': round(rotation, 2),
            'linesFound': lines_count,
            'originalWidth': int(original_width),
            'originalHeight': int(original_height),
            'debug': f'Линий найдено: {lines_count}, Угол: {round(rotation, 2)}°'
        })
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("🚀 Запуск сервиса определения угла...")
    print(" URL: http://localhost:5000")
    print(" Эндпоинт: POST /detect-angle")
    app.run(host='0.0.0.0', port=5000, debug=False)
