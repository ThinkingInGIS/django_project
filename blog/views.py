from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
# Create your views here.

from django.http import HttpResponse

@csrf_exempt
def nlp_page(request):
    result = None
    if request.method == "POST":
        text = request.POST.get('text', '')
         
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        # ======================
        # 模型路径（改成你保存的目录）
        # ======================
        MODEL_DIR = r"D:/PythonProjects/nlp/bert_cls_jd_cpu"
        # 加载 tokenizer 和 model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to("cpu")
        model.eval()

        def predict(text: str):
            enc = tokenizer(
                text,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=500
            )
            with torch.no_grad():
                outputs = model(**enc)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                pred_idx = int(torch.argmax(probs, dim=-1).item())
                pred_score = pred_idx + 1  # 映射回原始标签 (1..K)
                pred_conf = float(probs[0, pred_idx].item())
            return pred_score, pred_conf
        ##
        score, conf = predict(text)
        result = f"预测类别: {score}，置信度: {conf:.4f}"
        
    return render(request, "nlp_page.html", {"result": result})