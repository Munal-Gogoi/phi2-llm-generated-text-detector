---
base_model: microsoft/phi-2
library_name: peft
---

# Phi-2 Text Detector

A fine-tuned Phi-2 model that classifies text as human-written or AI-generated, with reasoning for each prediction.

## Model Details

### Model Description
- **Developed by:** Munal Gogoi
- **Model type:** Causal Language Model with PEFT (LoRA)
- **Language(s):** English
- **License:** MIT
- **Finetuned from:** microsoft/phi-2 (2.7B parameter Transformer)

### Model Sources
- **Repository:** https://github.com/Munal-Gogoi/phi2-llm-generated-text-detector
- **Base Model:** [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)

## Uses
This model can be used to classify a given text as either Human-generated or AI-generated, and provide a brief reasoning for its decision.

### Direct Use
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
"microsoft/phi-2",
trust_remote_code=True,
device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

inputs = tokenizer('''Classify as human or AI-generated:
Text: "I enjoy gardening on weekends"
Answer:''', return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs))


### Out-of-Scope Use
- Medical, legal, or financial decision-making
- Generating factual reports without verification
- Languages other than English

## Bias, Risks, and Limitations
- **Inaccuracy:** May produce incorrect classifications (inherited from Phi-2)
- **Societal biases:** Reflects biases in training data
- **Verbosity:** Tends toward wordy explanations
- **Code knowledge:** Limited to Python-based training data

### Recommendations
- Verify critical classifications manually
- Use thresholding for confidence scores
- Monitor for toxicity/bias in sensitive applications

## Training Details
### Training Data
- 800k+ text samples from human and AI sources (Bloom, GPT, Claude, etc.)
- Labels: Human=0, AI-generated=1

### Training Procedure
- **Method:** Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- **Quantization:** 4-bit (NF4)
- **Hyperparameters:**
  - Learning Rate: 1e-4
  - Batch Size: 2
  - Epochs: 1
  - Lora Config:
    - r=8
    - target_modules=["Wqkv", "fc1", "fc2"]
    - lora_alpha=32

## Evaluation
| Metric       | Value |
|--------------|-------|
| Accuracy     | 0.87  |
| Precision    | 0.89  |
| Recall       | 0.85  |
| F1           | 0.87  |

## Environmental Impact
- **Hardware:** 1x NVIDIA GPU (e.g., RTX A6000)
- **Training Time:** ~2 hours
- **CO2 Emissions:** ~0.8 kg (estimated via [ML CO2 calculator](https://mlco2.github.io/impact))

## Technical Specifications
- **Architecture:** Transformer with 2.7B parameters
- **Fine-tuning:** LoRA adapters (~0.1% trainable parameters)
- **Context Window:** 2048 tokens

## Citation
@misc{phi-2,
title={Phi-2: The surprising power of small language models},
author={Microsoft},
year={2023},
url={https://aka.ms/phi2}
}


## Model Card Contact
22051527@kiit.ac.in