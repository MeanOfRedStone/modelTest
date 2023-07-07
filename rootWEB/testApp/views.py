from django.shortcuts import render

# Create your views here.
def index(request):

    return render(request, 'index.html')


def transmit(request):
    story = request.POST['story']

    print(">>>>>>debgug , 사연 전송 확인 : ", story)
    return render(request, 'transmit.html')

