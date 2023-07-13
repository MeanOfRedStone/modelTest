import time

import torch as torch


from django.shortcuts import render
from transformers import BertTokenizerFast, BertModel


# Create your views here.
def index(request):

    return render(request, 'index.html')


def transmit(request):
    story = request.POST['story']
    
    print(">>>>>>debgug , 사연 전송 확인 : ", story)

    from ratsnlp.nlpbook.classification import ClassificationDeployArguments
    args = ClassificationDeployArguments(
        pretrained_model_name="beomi/kcbert-base",
        downstream_model_dir="model/kcbert",
        max_seq_length=128,
    )

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_name,
        do_lower_case=False,
    )

    import torch
    fine_tuned_model_ckpt = torch.load(
        args.downstream_model_checkpoint_fpath,
        map_location=torch.device("cpu"),
    )

    from transformers import BertConfig
    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_name,
        num_labels=fine_tuned_model_ckpt["state_dict"]["model.classifier.bias"].shape.numel(),
    )

    from transformers import BertForSequenceClassification
    model = BertForSequenceClassification(pretrained_model_config)

    model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})

    model.eval()

    inputs = tokenizer(
        [story],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
    )
    with torch.no_grad():
        outputs = model(**{k: torch.tensor(v) for k, v in inputs.items()})
        prob = outputs.logits.softmax(dim=1)
        positive_prob = round(prob[0][1].item(), 4)
        negative_prob = round(prob[0][0].item(), 4)
        pred = "긍정 (positive)" if torch.argmax(prob) == 1 else "부정(negative)"

    print(">>>>>> debug, 판단 ", pred)
    print(">>>>>> debug, 긍정 판단 확률", positive_prob)
    print(">>>>>> debug, 부정 판단 확률", negative_prob)




    return render(request, 'transmit.html')

