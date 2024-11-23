# text = """Dear Amazon, last week I ordered an Optimus Prime action figure
# from your online store in Germany. Unfortunately, when I opened the package.
# I discovered to my horror that I had been sent an action figure of Megatron
# instead! As a lifelong enemy of the Decepticons, I hope you can understand my
# dilemma. To resolve the issue, I demand an exchange of Megatron for the
# Optimus Prime figure I ordered. Enclosed are copies of my records concerning
# this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""
#
#
#
#
#
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)
#
# from transformers import pipeline
#
# # 初始化 pipeline
# classifier = pipeline(
#     "sentiment-analysis",
#     model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
#     device=0  # 使用 GPU，如果没有 GPU，则设置为 -1
# )
#
# # 测试
# result = classifier(text)
# print(result)
#
#




## applied for Chinese analysis

from transformers import BertTokenizer, BertForSequenceClassification, pipeline

model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

result = classifier("这是一个使用中文BERT模型的例子。")
print(result)
