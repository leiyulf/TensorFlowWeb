import json
import os
import pandas as pd
import tensorflow as tf
import threading
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import losses, optimizers, activations
from tensorflow.keras.callbacks import Callback
from django.conf import settings

# 自定义回调
class CustomCallback(Callback):
    def __init__(self, modelName):
        super(CustomCallback, self).__init__()
        self.modelName = modelName
    def on_epoch_end(self, epoch, logs=None):
        log_file_path = os.path.join(settings.MODEL_LOG_PATH, f"{self.modelName}.txt")
        with open(log_file_path, 'a') as f:
            trainState = "training"
            if (epoch + 1) == self.params['epochs']:
                trainState = "finish"
            f.write(f"{trainState}$${epoch + 1}$${self.params['epochs']}$${logs['loss']:.4f}$${logs['val_loss']:.4f}$${settings.MODEL_FILE_PATH}\\{self.modelName}\n")
            # 训练损失（Training Loss） 验证损失（Validation Loss）

class TrainingThread(threading.Thread):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            modelStructure = self.params.get('modelStructure');  # 模型结构
            modelConfig = self.params.get('modelConfig');  # 模型配置
            csvPath = self.params.get('csvPath');  # 训练集路径
            modelName = self.params.get('modelName');  # 模型路径

            characteristicVar = modelConfig.get('characteristicVar');  # 特征变量
            targetVar = modelConfig.get('targetVar');  # 目标变量
            optimizer = modelConfig.get('optimizer');  # 优化器
            lossFunction = modelConfig.get('lossFunction');  # 损失函数
            periodization = modelConfig.get('periodization');  # 训练周期
            batchSize = modelConfig.get('batchSize');  # 批次大小

            # 开始训练
            data = pd.read_csv(csvPath)
            target_variables = targetVar
            selected_features = characteristicVar
            X = data[selected_features]
            Y = data[target_variables]

            # 数据标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)  # 使用StandardScaler标准化特征

            # 划分训练集和测试集
            X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

            # 定义MLP模型
            model = tf.keras.Sequential()
            for modelCell in modelStructure:
                activationFunctionValue = None
                activationFunction = modelCell.get('activationFunction', "")
                cellName = modelCell.get('cellName', "")
                neuronFormat = modelCell.get('neuronFormat', "")

                if activationFunction == "relu":
                    activationFunctionValue = activations.relu
                elif activationFunction == "sigmoid":
                    activationFunctionValue = activations.sigmoid
                elif activationFunction == "tanh":
                    activationFunctionValue = activations.tanh
                elif activationFunction == "softmax":
                    activationFunctionValue = activations.softmax

                if "输入层" == cellName:
                    model.add(tf.keras.layers.Dense(neuronFormat, activation=activationFunctionValue,
                                                    input_shape=(X_train.shape[1],)))
                elif "隐藏层" == cellName:
                    model.add(tf.keras.layers.Dense(neuronFormat, activation=activationFunctionValue))
                elif "输出层" == cellName:
                    model.add(tf.keras.layers.Dense(neuronFormat))

            # 编译模型
            optimizerValue = None
            lossValue = None
            if optimizer == 'SGD':
                # learning_rate： 学习率，表示每次参数更新的步长。在你的代码中，学习率被设置为0.01，这是一个常见的初始值。你可以根据实际情况调整学习率，通常需要进行实验来找到最佳值。
                # momentum： 动量，是一个在0和1之间的浮点数，控制着过去梯度对当前梯度的影响。使用动量有助于在训练过程中克服局部最小值。在你的代码中，动量被设置为0.9，这也是一个常见的初始值。
                optimizerValue = optimizers.SGD(learning_rate=0.01, momentum=0.9)
            elif optimizer == 'Adam':
                # learning_rate： 学习率，表示每次参数更新的步长。它是一个正数，通常在0.1、0.01、0.001等范围内选择。学习率越小，参数更新的幅度越小，训练过程可能更加稳定，但训练时间可能更长。
                # beta_1： 一阶矩估计的衰减系数。它是一个在0和1之间的浮点数，通常接近于1。它控制了一阶矩估计（梯度的平均值）的衰减速度。
                # beta_2： 二阶矩估计的衰减系数。同样是一个在0和1之间的浮点数，通常接近于1。它控制了二阶矩估计（梯度平方的平均值）的衰减速度。
                # epsilon： 一个很小的常数，通常在数值上接近于零。它用于防止除零错误，添加到分母中。
                optimizerValue = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
            elif optimizer == 'RMSprop':
                # learning_rate： 学习率，表示每次参数更新的步长。在你的代码中，学习率被设置为0.001，这是一个常见的初始值。你可以根据实际情况调整学习率。
                # rho： 一个在0和1之间的浮点数，用于控制平方梯度的移动平均的衰减率。rho的默认值通常为0.9，这也是你在代码中设置的值。它影响了对梯度平方的估计。
                optimizerValue = optimizers.RMSprop(learning_rate=0.001, rho=0.9)
            elif optimizer == 'Adagrad':
                optimizerValue = optimizers.Adagrad(learning_rate=0.01)

            if lossFunction == 'mean_squared_error':
                lossValue = losses.MeanSquaredError()
            elif lossFunction == 'mean_absolute_error':
                lossValue = losses.MeanAbsoluteError()
            elif lossFunction == 'huber_loss':
                lossValue = losses.Huber()

            model.compile(optimizer=optimizerValue, loss=lossValue)

            # 训练模型
            model.fit(X_train, Y_train, epochs=periodization, batch_size=batchSize, validation_data=(X_test, Y_test),
                      callbacks=[CustomCallback(modelName)])

            # 评估模型
            loss = model.evaluate(X_test, Y_test)  # 评估模型在测试集上的性能
            print(f'Mean Squared Error on Test Set: {loss}')
            # 将训练好的模型保存到指定目录
            model.save(os.path.join(settings.MODEL_FILE_PATH, modelName))
            print(f'over...')
        except Exception as e:
            print(f'An error occurred during training: {str(e)}')


@csrf_exempt
def TrainingModel(request):
    if request.method == 'POST':
        try:
            paramsJson = json.loads(request.body)
            thread = TrainingThread(paramsJson)
            thread.start()
            return JsonResponse({'message': '模型开始训练', 'success': True}, status=200)
        except json.JSONDecodeError as e:
            return JsonResponse({'message': 'Invalid JSON data', 'success': False}, status=400)
        except Exception as e:
            return JsonResponse({'message': f'An error occurred: {str(e)}', 'success': False}, status=500)
    else:
        data = {'message': 'unsupported request method'}
        return JsonResponse(data, status=405)
