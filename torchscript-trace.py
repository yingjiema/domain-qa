import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_save_name = 'model.pt'
model_name = 'dmis-lab/biobert-base-cased-v1.1-squad'

model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer.encode('What type of cell is a neutrophil?', return_tensors='pt',max_length=256,truncation=True,padding='max_length')

class PyToScript(torch.nn.Module):
    def __init__(self,model):
        super(PyToScript,self).__init__()
        self.model=model
    def forward(self,data):
        return self.model(data).end_logits

pt_model=PyToScript(model).eval()
traced=torch.jit.trace(pt_model,inputs)
traced.save(model_save_name)