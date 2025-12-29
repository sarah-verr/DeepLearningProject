from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests
import torch

model_id = "xtuner/llava-phi-3-mini-hf"

# 1. Load Model
# Note: trust_remote_code=True might be needed for some Phi-3 versions
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto", 
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# 2. Prepare Image and Prompt
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Note: Phi-3 uses a specific prompt format
prompt = "<|user|>\n<image>\nWhat is this image about?<|end|>\n<|assistant|>\n"

inputs = processor(prompt, image, return_tensors="pt").to("cuda")

# 3. Generate
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))