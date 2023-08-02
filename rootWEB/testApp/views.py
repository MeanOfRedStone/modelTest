import time
import torch as torch
from django.shortcuts import render, redirect
from rest_framework import permissions, status
from rest_framework.decorators import api_view, permission_classes
from transformers import BertTokenizerFast, BertModel, BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#장고 레스트 프레임워크
from testApp.models import ModelResult
from testApp.serializers import ModelResultSerializer
from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
from rest_framework.response import Response


# Create your views here.
def index(request):

    return render(request, 'index.html')


@api_view(['GET', 'POST'])
@permission_classes((permissions.AllowAny,))
def transmit(request):
    # story = request.POST['story']
    story = request.data['story']
    print(">>>>>>debgug , 사연 전송 확인 : ", story)


    """
    현재 KcBERT로 긍정, 부정을 먼저 판별해준다.
    추후 기본 BERT의 성능이 더 좋을 경우 아래 값은 바뀔수도 있음
    """

    #불러와서 사용할 모델 속성 값 설정
    from ratsnlp.nlpbook.classification import ClassificationDeployArguments
    args = ClassificationDeployArguments(
        pretrained_model_name="beomi/kcbert-base",
        downstream_model_dir="model/kcbert",
        max_seq_length=128,
    )

    #버트 토크나이저 속성 값 설정
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_name,
        do_lower_case=False,
    )

    #모델을 cpu 환경에 불러온다
    import torch
    fine_tuned_model_ckpt = torch.load(
        args.downstream_model_checkpoint_fpath,
        map_location=torch.device("cpu"),
    )

    #훈련시킨 모델의 속성값을 읽어들임
    from transformers import BertConfig
    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_name,
        num_labels=fine_tuned_model_ckpt["state_dict"]["model.classifier.bias"].shape.numel(),
    )

    #속성 값에 맞게 모델을 생성
    from transformers import BertForSequenceClassification
    model = BertForSequenceClassification(pretrained_model_config)

    model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})

    model.eval()

    #전송한 사연을 토크나이저에 넣어주는 부분 [story]값을 알맞은 input값으로 바꿔서 사용
    inputs = tokenizer(
        [story],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
    )

    #사연을 바탕으로 평가하는 부분
    #1일 때 긍정 0일 때 부정임
    with torch.no_grad():
        outputs = model(**{k: torch.tensor(v) for k, v in inputs.items()})
        prob = outputs.logits.softmax(dim=1)
        positive_prob = round(prob[0][1].item(), 4)
        negative_prob = round(prob[0][0].item(), 4)
        pred = 1 if torch.argmax(prob) == 1 else 0

    #debugging용
    print(">>>>>> debug, 1차 판단 긍정 판단 확률", positive_prob)
    print(">>>>>> debug, 1차 판단 부정 판단 확률", negative_prob)

    #모델 이원화에 따라 긍정으로 판단할 경우 / 부정으로 판단할 경우를 나누어 세부 감정 모델로 한 번 더 판단

    if pred == 1:
        """
        긍정으로 나올 경우 세부 데이터의 부재로 사용자에게 세부 감정 판별을 맡김
        """


        pred = "pos"

        print(">>>>>>debug, 최종 결과- pred : ", pred)

        #API생성을 위해서 판별된 결과값을 ModelResult에 저장
        ModelResult(story=story, prediction=pred).save()

        # url: predict/ 로 redirect해 api바로 연결
        return redirect('api')
    else :

        """
        부정으로 나올 경우 Anger와 Sadness를 구분
        """

        """
        < 코드요약 >
        1.
        모델 불러오기
        프로젝트 rootWEB에 있는 model 폴더에서 bert_pos.pt를 호출

        2. 토크나이저 생성
        자연어를 기계어로 처리하기 위한 Bert토크나이저 생성

        3. 토큰화
        '2'에서 생성한 토크나이저로 받아온 자연어를 컴퓨터가 이해할 수 있도록 토큰화 시키는 부분
        """
        # 1. 모델 불러오기
        device = torch.device("cpu")
        PATH = 'model/bert/bert_neg.pt'
        model = torch.load(PATH, map_location=device)
        # print(">>>>>>debug, model 확인 : ", model)

        # 2. 토크나이저 생성
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

        # 3. 토큰화
        # 1) def BertToknizer 부분
        # (1)add special tokens
        sentences = ["[CLS]" + str(sent) + "[SEP]" for sent in story]
        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
        result_tokenized_texts = []
        # (2)encode into integers
        for text in tokenized_texts:
            encoded_sent = tokenizer.convert_tokens_to_ids(text)
            result_tokenized_texts.append(encoded_sent)

        # 생성한 'LIST' result를 아래에서 활용

        # 2) def convert_data 부분
        # tokenize
        tokenized_sent = result_tokenized_texts
        # pad sentnece
        data = pad_sequences(tokenized_sent, maxlen=128, dtype='long', truncating='post', padding='post')
        padded_sent = data

        # 3) def create_masks 부분
        # create atteintion mask
        masks = []
        for sent in padded_sent:
            mask = [float(s > 0) for s in sent]
            masks.append(mask)

        masks_sent = masks

        # 4) torch tensor
        inputs = torch.tensor(padded_sent)
        masks = torch.tensor(masks_sent)

        # 4. 토큰을 모델에 투입해 판별
        # 1) model to evaluation model
        model.eval()

        # 2)
        # predict data

        b_inputs_ids = inputs.to(device)
        b_input_mask = masks.to(device)

        with torch.no_grad():
            outputs = model(b_inputs_ids, token_type_ids=None, attention_mask=b_input_mask)
            # get prediction
            logits = outputs[0]
            prob = logits.softmax(dim=1)

            #negative_0은 '부정감정2' , Anger
            negative_0 = round(prob[0][0].item(), 4)
            print(">>>>>>debug, negative_0(Anger) 확률값 : ", negative_0)

            #negative_1은 "부정감정3' , Sadness
            negative_1 = round(prob[0][1].item(), 4)
            print(">>>>>>debug, negative_1(Sadness) 확률값 : ", negative_1)
            if (negative_0 > negative_1):
                #0의 확률값이 더 클 경우(Anger)를 반환
                pred = "ang"
            else:
                #1의 확률값이 더 클 경우(Sadness) 반환
                pred = "sad"
        print(">>>>>>debug, 최종 결과- pred : ", pred)

        print("부정")

        ModelResult(story=story, prediction=pred).save()

        """
            List all code snippets, or create a new snippet.
            """
        if request.method == 'GET':
            prediction = ModelResult.objects.all()
            serializer = ModelResultSerializer(prediction, many=True)
            return Response(serializer.data)

        elif request.method == 'POST':
            serializer = ModelResultSerializer(data=request.data)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


        #url: predict/ 로 redirect해 api바로 연결
        # return render('api')

@api_view(['GET', 'POST'])
@permission_classes((permissions.AllowAny,))
def prediction_list(request, format=None):
    """
    List all code snippets, or create a new snippet.
    """
    if request.method == 'GET':
        prediction = ModelResult.objects.all()
        serializer = ModelResultSerializer(prediction, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = ModelResultSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'PUT', 'DELETE'])
@permission_classes((permissions.AllowAny,))
def prediction_detail(request, pk, format=None):
    """
    Retrieve, update or delete a code snippet.
    """
    try:
        prediction = ModelResult.objects.get(pk=pk)
    except ModelResult.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer = ModelResultSerializer(prediction)
        return Response(serializer.data)

    elif request.method == 'PUT':
        serializer = ModelResultSerializer(prediction, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        prediction.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

"""
참고용으로 남겨놓음
def transmit_pos(request):
    story_pos = request.POST['story_pos']

    print(">>>>>>debgug , 사연 전송 확인 : ", story_pos)

    
    <코드 요약>
    1. 모델 불러오기
    프로젝트 rootWEB에 있는 model 폴더에서 bert_pos.pt를 호출
    
    2. 토크나이저 생성
    자연어를 기계어로 처리하기 위한 Bert토크나이저 생성
    
    3. 토큰화
    '2'에서 생성한 토크나이저로 받아온 문장을 토큰화 시키는 부분
    

    #1. 모델 불러오기
    device = torch.device("cpu")
    PATH = 'model/bert/bert_pos.pt'
    model = torch.load(PATH, map_location=device)
    # print(">>>>>>debug, model 확인 : ", model)

    #2. 토크나이저 생성
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

    #3. 토큰화
    #1) def BertToknizer 부분
    #(1)add special tokens
    sentences = ["[CLS]" + str(sent) + "[SEP]" for sent in story_pos]
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    result_tokenized_texts = []
    #(2)encode into integers
    for text in tokenized_texts:
        encoded_sent = tokenizer.convert_tokens_to_ids(text)
        result_tokenized_texts.append(encoded_sent)
    #여기서 생성한 리스트 result를 아래에서 활용

    #2) def convert_data 부분
    # tokenize
    tokenized_sent = result_tokenized_texts
    #pad sentnece
    data = pad_sequences(tokenized_sent, maxlen=128, dtype='long', truncating='post', padding='post')
    padded_sent = data

    #3) def create_masks 부분
    # create atteintion mask
    masks = []
    for sent in padded_sent:
        mask = [float(s > 0) for s in sent]
        masks.append(mask)

    masks_sent = masks

    # 4) torch tensor
    inputs = torch.tensor(padded_sent)
    masks = torch.tensor(masks_sent)


    #4. 토큰을 모델에 투입해 판별
    #1) model to evaluation model
    model.eval()

    #2)
    # predict data

    b_inputs_ids = inputs.to(device)
    b_input_mask = masks.to(device)

    with torch.no_grad():
        outputs = model(b_inputs_ids, token_type_ids=None, attention_mask=b_input_mask)
        # print(">>>>>>Debug, outputs: ", outputs)
        # get prediction
        logits = outputs[0]
        # print(">>>>>>Debug, outputs : ", type(logits))
        prob = logits.softmax(dim=1)
        # print(">>>>>>debug, prob : ", prob)

        positive_0 = round(prob[0][0].item(), 4)
        # print(type(positive_0))
        # positive_0 = int(positive_0)
        # print(">>>>>>debug, positive_0 : ", positive_0)
        positive_1 = round(prob[0][1].item(), 4)
        # positive_1 = int(positive_1)
        # print(type(positive_1))
        # print(">>>>>>debug, positive_1 : ", positive_1)
        if (positive_0 > positive_1):
            pred = 0
        else:
            pred = 1
        # print(">>>>>>debug, pred : ", pred)
        # positive_2 = round(prob[0][2].item(), 4)
        # print(">>>>>>debug, positive_2 : ", positive_2)
        # positive_3 = round(prob[0][3].item(), 4)
        # print(">>>>>>debug, positive_3 : ", positive_3)
        # positive_4 = round(prob[0][4].item(), 4)
        # print(">>>>>>debug, positive_4 : ", positive_4)
        # pred = np.argmax(logits)
        # print(">>>>>>Debug, pred : ", logits.numpy())
        # print(">>>>>>debug, pred : ",pred)

    print(">>>>>>debug, 최종 결과- pred : ", pred)
    return render(request, 'transmit_pos.html')
"""

ModelResult.objects.all().delete()