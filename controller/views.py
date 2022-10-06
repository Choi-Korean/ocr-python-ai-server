import json
import os
from django.http import JsonResponse
from django.shortcuts import render

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny

import cv2
import numpy as np
from PIL import Image

from ai.settings import BASE_DIR
from .apps import ControllerConfig
from .models import Requested, Responsed
from .serializers import RequestedSerializer, ResponsedSerializer


class GetInformation(APIView):
    permission_classes = [AllowAny]
    serializer_class = RequestedSerializer

    def post(self, request, format=None):
        serializer = self.serializer_class(data=request.data)  # data 유효한지 확인
        if serializer.is_valid():
            image = request.data.get('image')
            user = user = serializer.data.get('user')
            req = Requested(user=user,
                            image=image
                            )
            file_name, file_ext = os.path.splitext(image.name)
            sliced_img_names = []
            sliced_img_list = {}
            # req.save()

            # Super Resolution 초해상화

            # Image Classification

            # Text Detection
            # 현재는 AI Hub Label 데이터 추출해서 Detection 하는 형식으로
            # 추출 -> static/images에 영역별로 분리해서 저장
            with open(os.path.join(BASE_DIR, 'static', 'json', file_name + '.json'), 'r', encoding='utf8') as f:
                json_files = json.load(f)
            img_info = json_files['images'][0]

            # Inmemoryuploadedfile 을 이미지로 읽기
            img = cv2.imdecode(np.fromstring(
                image.read(), np.uint8), cv2.IMREAD_COLOR)
            # img = cv2.imread(image, cv2.IMREAD_COLOR)
            targets = json_files['annotations'][0]['polygons']

            for i, l in enumerate(targets):
                if l['type'] != int(2):
                    continue
                d1 = list(map(int, l['points'][0]))
                d2 = list(map(int, l['points'][2]))
                try:
                    sliced_img = img[d1[1]:d2[1], d1[0]:d2[0]]  # 높이, 너비
                    sliced_img_name = img_info['identifier'] + \
                        f'_{i}.' + img_info['type']

                    sliced_img_list[sliced_img_name] = [
                        sliced_img, l['points']]
                    sliced_img_names.append(sliced_img_name)
                    # 저장 안하고, 바로 text recognition으로 보내기
                    # cv2.imwrite(os.path.join(BASE_DIR, 'static',
                    #             'images', image, sliced_img_name), sliced_img)
                except:
                    pass

            # # Text Recognition
            result = [{'name': file_name + file_ext, 'result': []}]
            for k in sliced_img_names:
                # ndarray -> 이미지로 다시 전환해서 바로 보내기. (서버에 나눈 이미지 저장 후 불러오는게 X)
                pred, confidence_score = ControllerConfig.tr.predict(
                    k, image=[Image.fromarray(sliced_img_list[k][0])])
                result[0]['result'].append(
                    [pred, confidence_score, sliced_img_list[k][1]])

            res = Responsed(user=user, result=result)
            # res = Responsed(user=user, result=json.dumps(result))

            return Response(ResponsedSerializer(res).data, status=status.HTTP_200_OK)
        return Response({'Bad Request': 'Invalid Data..'}, status=status.HTTP_400_BAD_REQUEST)
