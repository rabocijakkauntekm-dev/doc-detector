from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import math

app = Flask(__name__)


def detect_angle_hough(image):
    """
    Определяет угол поворота по линиям текста.
    Фильтрует шум, берёт только длинные горизонтальные линии.
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


def detect_text_bounds(image):
    """
    Находит границы текста на изображении.
    Работает с белым документом на тёмном/светлом фоне.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    
    # Метод 1: Инвертированный порог (тёмный текст на светлом фоне)
    _, binary_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Метод 2: Прямой порог (светлый документ на тёмном фоне)
    _, binary_direct = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Объединяем
    binary = cv2.bitwise_or(binary_inv, binary_direct)
    
    # Морфология
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=5)
    eroded = cv2.erode(dilated, kernel, iterations=4)
    
    # Контуры
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Сортируем по площади
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Ищем контур документа (большой, но не всё изображение)
    doc_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        # Пропускаем слишком маленькие и слишком большие
        if area > width * height * 0.1 and area < width * height * 0.95:
            doc_contour = contour
            break
    
    # Если нашли контур документа
    if doc_contour is not None:
        x, y, w, h = cv2.boundingRect(doc_contour)
        
        padding = 20
        crop_x = max(0, x - padding)
        crop_y = max(0, y - padding)
        crop_w = min(w + padding * 2, width - crop_x)
        crop_h = min(h + padding * 2, height - crop_y)
        
        return crop_x, crop_y, crop_w, crop_h, len(contours), 'contour'
    
    # Если не нашли — используем центр изображения (80%)
    margin_x = int(width * 0.1)
    margin_y = int(height * 0.1)
    return margin_x, margin_y, width - margin_x * 2, height - margin_y * 2, len(contours), 'center'


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
        rotation, lines_count = detect_angle_hough(image)
        
        # 2. Находим границы текста
        crop_x, crop_y, crop_w, crop_h, contours_count, method = detect_text_bounds(image)
        
        return jsonify({
            'rotation': round(rotation, 2),
            'cropX': int(crop_x),
            'cropY': int(crop_y),
            'cropWidth': int(crop_w),
            'cropHeight': int(crop_h),
            'originalWidth': int(original_width),
            'originalHeight': int(original_height),
            'debug': f'Линий: {lines_count}, Контуров: {contours_count}, Метод: {method}, Угол: {round(rotation, 2)}°',
            'bounds': f'{crop_x},{crop_y},{crop_w},{crop_h}'
        })
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("🚀 Запуск сервиса детекции документов...")
    print(" URL: http://localhost:5000")
    print(" Эндпоинт: POST /detect")
    app.run(host='0.0.0.0', port=5000, debug=False)
