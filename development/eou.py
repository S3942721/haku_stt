import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# Download and load tokenizer and model
model_id = "latishab/turnsense"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_path = hf_hub_download(repo_id=model_id, filename="model_quantized.onnx")

# Initialize ONNX Runtime session
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# Prepare input
text = "Hello, how are you?"
inputs = tokenizer(
    f"<|user|> {text} <|im_end|>",
    padding="max_length",
    max_length=256,
    return_tensors="pt"
)

# Run inference
ort_inputs = {
    'input_ids': inputs['input_ids'].numpy(),
    'attention_mask': inputs['attention_mask'].numpy()
}
probabilities = session.run(None, ort_inputs)[0]
