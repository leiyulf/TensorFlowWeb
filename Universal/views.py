from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.http import StreamingHttpResponse
import asyncio

@csrf_exempt
def GenerateTrainData(request):
    if request.method == 'POST':
        data = {'message': 'hello'}
        return JsonResponse(data, status=200)
    else:
        data = {'message': 'unsupported request method'}
        return JsonResponse(data, status=405)

async def sse(request):
     async def event_stream():
        while True:
            # 从数据库或其他数据源获取数据
            data = "1"
            # 构造SSE消息
            event = 'message'
            message = f'data: {data}\n\n'
            yield f'event: {event}\n{message}'
            await asyncio.sleep(1)

     return StreamingHttpResponse(event_stream(), content_type='text/event-stream')
