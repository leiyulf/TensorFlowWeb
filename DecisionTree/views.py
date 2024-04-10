import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.model_selection import train_test_split
from django.conf import settings
import tensorflow as tf
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


@csrf_exempt
def TrainingModel(request):
    if request.method == 'POST':
        try:
            targetCode = None
            csvCol = [];
            paramsJson = json.loads(request.body)
            tableStructure = paramsJson.get('tableStructure')
            dataPath = paramsJson.get('dataPath')

            # 遍历获取结构
            for row in tableStructure:
                targetItem = row.get('targetItem')
                colCode = row.get('colCode')
                csvCol.append(colCode)
                if targetItem == "是":
                    targetCode = colCode
            df = pd.read_csv(dataPath)
            X = df.drop(targetCode, axis=1)
            Y = df[targetCode]
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            clf = DecisionTreeClassifier(random_state=42)
            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)
            accuracy = clf.score(x_test, y_test)
            print(accuracy)

            data = {'message': accuracy}
            return JsonResponse(data, status=200)
        except json.JSONDecodeError as e:
            data = {'message': 'Invalid JSON data'}
            return JsonResponse(data, status=400)
        except Exception as e:
            data = {'message': f'An error occurred: {str(e)}'}
            return JsonResponse(data, status=500)
    else:
        data = {'message': 'unsupported request method'}
        return JsonResponse(data, status=405)
