import numpy as np
import cv2
import base64
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.models import load_model
import json

# 加载模型
filePath = r'D:\workspace\CODE\Django\TensorflowWeb\cnn\model\number_resnet.h5'
model = load_model(filePath)


@csrf_exempt
def ModelPrediction(request):
    if request.method == 'POST':
        paramsJson = json.loads(request.body)
        if 'images' in paramsJson:
            # 获取 base64 图片列表
            images_base64 = paramsJson.get('images')

            # 存储预测结果
            predictions = []

            # 对每张图片进行预测
            for image_base64 in images_base64:
                try:
                    while len(image_base64) % 4 != 0:
                        image_base64 += '='
                    # 解码 base64 图片
                    image_data = base64.b64decode(image_base64)

                    # 将图像数据转换为OpenCV格式
                    nparr = np.frombuffer(image_data, np.uint8)
                    image_cv2 = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                    # 图像处理和预测
                    prediction = process_and_predict(image_cv2)

                    # 添加预测结果到列表
                    predictions.append({'image': image_base64, 'prediction': prediction})
                except Exception as e:
                    print(f"Error processing image: {e}")

            # 返回预测结果
            return JsonResponse({'predictions': predictions}, status=200)
        else:
            data = {'message': 'images not found in POST request'}
            return JsonResponse(data, status=400)
    else:
        data = {'message': 'unsupported request method'}
        return JsonResponse(data, status=405)


def process_and_predict(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
    # 自适应阈值
    binary_adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 9);

    # 轮廓提取
    contours, _ = cv2.findContours(binary_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 对轮廓按 x 坐标进行排序
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    predictions = ""
    for contour in contours:
        # 获取数字的边界框
        x, y, w, h = cv2.boundingRect(contour)
        if w < 4 and h < 4:
            continue
        # 提取
        digit = binary_adaptive[y:y + h, x:x + w]

        canvas = np.zeros((28, 28), dtype=np.uint8)

        # 将数字调整为 10 大小
        resized_digit = cv2.resize(digit, (10, 18))

        # 计算将数字放置在画布中央的位置
        canvas_h, canvas_w = canvas.shape
        digit_h, digit_w = resized_digit.shape
        start_h = (canvas_h - digit_h) // 2
        start_w = (canvas_w - digit_w) // 2

        # 将数字放置在画布中央
        canvas[start_h:start_h + digit_h, start_w:start_w + digit_w] = resized_digit

        # 进行预处理，将像素值映射到[0, 1]
        normalized_digit = canvas / 255.0

        # 展平并添加批处理维度
        flattened_digit = normalized_digit.reshape(1, 28, 28, 1)

        # 进行预测
        prediction = model.predict(flattened_digit)

        # 获取预测结果
        predicted_number = np.argmax(prediction)
        predictions += str(predicted_number)
    return predictions