Optimized GPT-2 Implementation
Overview
This repository contains an optimized implementation of the GPT-2 model using PyTorch. The model architecture is designed to be efficient and modular, allowing for easy customization and integration into various applications. The implementation supports loading pretrained weights from Hugging Face's Transformers library.
Features
Modular Design: The code is organized into distinct classes for different components of the model, including attention mechanisms, feed-forward networks, and layer normalization.
Pretrained Model Loading: Easily load pretrained GPT-2 weights from Hugging Face's model hub.
Configurable Architecture: Utilize a configuration class to easily modify the number of layers, heads, embedding size, vocabulary size, and block size.
Installation
To get started, clone the repository and install the required dependencies:
bash
git clone https://github.com/srirangaakhil/optimized-gpt2.git
cd optimized-gpt2
pip install torch transformers

Usage
Importing the Model
You can import the model in your Python script as follows:
python
import torch
from model import GPT2, modelconf

# Load the pretrained GPT-2 model
model = GPT2.from_pretrained('gpt2')
print('Model loaded successfully')

Configuration
The model configuration can be adjusted by modifying the parameters in the modelconf class:
python
conf = modelconf(n_layer=12, n_head=12, n_embd=768)
model = GPT2(conf)

Forward Pass
To perform a forward pass through the model with input data:
python
# Sample input tensor (Batch size x Sequence length x Embedding size)
input_tensor = torch.randn(1, 1024, 768)  # Example shape

# Forward pass
output = model(input_tensor)
print(output.shape)  # Should be (1, 1024, 50257)

Model Architecture
The architecture consists of several key components:
Attention Mechanism: Implements multi-head attention with scaling and masking.
Feed-Forward Network (FFN): A two-layer feed-forward network with GELU activation.
Layer Normalization: Applied before each sub-layer to stabilize training.

Contributing
Contributions are welcome! If you have suggestions or improvements, please create a pull request or open an issue.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
This implementation is inspired by the original GPT-2 paper and utilizes code from Hugging Face's Transformers library for loading pretrained models.
