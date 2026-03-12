from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import math

app = Flask(__name__)


def detect_angle_hough(image):
    """
    Определяет угол поворота по линиям текста.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Размытие
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Детекция краев
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Hough Lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=100, maxLineGap=10)
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            dx = x2 - x1
            dy = y2 - y1
            
            length = math.sqrt(dx * dx + dy * dy)
            if length < 80:
                continue
            
            if abs(dx) < 30:
                continue
            
            angle = math.degrees(math.atan2(dy, dx))
            
            if abs(angle) > 45:
                angle = angle - 90 if angle > 0 else angle + 90
            
            if abs(angle) < 30:
                angles.append(angle)
    
    if len(angles) > 0:
        avg_angle = sum(angles) / len(angles)
        return -avg_angle, len(angles)
    
    return 0, 0


def detect_text_bounds(image):
    """
    Находит границы текста на изображении.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    
    # Адаптивный порог + Отсу
    binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.bitwise_or(binary_adaptive, binary_otsu)
    
    # Морфология
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=3)
    eroded = cv2.erode(dilated, kernel, iterations=2)
    
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    all_points = []
    for contour in contours[:10]:
        area = cv2.contourArea(contour)
        if area > 500 and area < width * height * 0.9:
            for point in contour:
                all_points.append(point[0])
    
    if len(all_points) > 0:
        all_points = np.array(all_points)
        x, y, w, h = cv2.boundingRect(all_points)
        
        padding = 30
        crop_x = max(0, x - padding)
        crop_y = max(0, y - padding)
        crop_w = min(w + padding * 2, width - crop_x)
        crop_h = min(h + padding * 2, height - crop_y)
        
        return crop_x, crop_y, crop_w, crop_h, len(contours), 'text'
    
    # Если текст не найден — ищем контур документа
    _, binary_inv = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary_inv, kernel, iterations=5)
    eroded = cv2.erode(dilated, kernel, iterations=4)
    
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > width * height * 0.1 and area < width * height * 0.95:
            x, y, w, h = cv2.boundingRect(contour)
            
            padding = 20
            crop_x = max(0, x - padding)
            crop_y = max(0, y - padding)
            crop_w = min(w + padding * 2, width - crop_x)
            crop_h = min(h + padding * 2, height - crop_y)
            
            return crop_x, crop_y, crop_w, crop_h, len(contours), 'document'
    
    margin_x = int(width * 0.05)
    margin_y = int(height * 0.05)
    return margin_x, margin_y, width - margin_x * 2, height - margin_y * 2, 0, 'center'


def rotate_and_crop(image, angle, crop_x, crop_y, crop_w, crop_h):
    """
    Поворачивает изображение и обрезает по координатам.
    Сначала поворот, потом crop по новым координатам.
    """
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    
    # Поворот изображения
    rotated = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotated, (width, height), 
                                    borderMode=cv2.BORDER_REPLICATE)
    
    # Обрезка
    cropped = rotated_image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    
    return cropped


@app.route('/detect', methods=['POST'])
def detect():
    """
    API для детекции угла и границ документа.
    Возвращает координаты + готовое обработанное изображение!
    """
    try:
        data = request.get_json()
        
        if not data or 'imageBase64' not in data:
            return jsonify({'error': 'imageBase64 не передан'}), 400
        
        image_base64 = data['imageBase64']
        
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_bytes = base64.b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Не удалось декодировать изображение'}), 400
        
        original_height, original_width = image.shape[:2]
        
        # 1. Определяем угол
        rotation, lines_count = detect_angle_hough(image)
        
        # 2. Находим границы
        crop_x, crop_y, crop_w, crop_h, contours_count, method = detect_text_bounds(image)
        
        # 3. Поворачиваем и обрезаем на сервере!
        processed_image = rotate_and_crop(image, rotation, crop_x, crop_y, crop_w, crop_h)
        
        # 4. Кодируем результат в base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'rotation': round(rotation, 2),
            'cropX': int(crop_x),
            'cropY': int(crop_y),
            'cropWidth': int(crop_w),
            'cropHeight': int(crop_h),
            'originalWidth': int(original_width),
            'originalHeight': int(original_height),
            'processedImage': f'data:image/jpeg;base64,{processed_base64}',
            'debug': f'Линий: {lines_count}, Контуров: {contours_count}, Метод: {method}, Угол: {round(rotation, 2)}°'
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
