from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.onnx import export
from transformers.onnx import FeaturesManager
from pathlib import Path

# Load your fine-tuned TinyBERT model
model_path = "D:/faizan/results/full_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define ONNX export path
onnx_path = Path("tinybert_onnx")
onnx_path.mkdir(exist_ok=True)

# Get ONNX configuration
model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature="sequence-classification")
onnx_config = model_onnx_config(model.config)

# Export model to ONNX
export(
    preprocessor=tokenizer,
    model=model,
    config=onnx_config,
    opset=17,
    output=onnx_path
)

print("✅ Export to ONNX complete!")
