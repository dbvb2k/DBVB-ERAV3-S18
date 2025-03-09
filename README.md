# Phi-2 Fine-tuned Chatbot 🤖

## Overview
This project implements a conversational AI chatbot using Microsoft's Phi-2 model, fine-tuned on the OpenAssistant dataset. The chatbot features a modern, user-friendly web interface built with Gradio and supports real-time conversation with memory capabilities.

## Features
- 🧠 Fine-tuned Phi-2 model for enhanced conversational abilities
- 💬 Interactive chat interface with message history
- 🎨 Modern, responsive UI with gradient styling
- ⚡ Real-time response generation
- 📝 Example prompts for easy testing
- 🔄 Conversation memory
- 🚀 Easy deployment options

## Technical Architecture
- **Base Model**: Microsoft Phi-2
- **Fine-tuning**: QLoRA (Quantized Low-Rank Adaptation)
- **Training Data**: OpenAssistant dataset
- **Frontend**: Gradio web interface
- **Memory**: Conversation history tracking
- **Optimization**: 4-bit quantization for efficient inference

## Installation

### Prerequisites
- Python 3.8 or higher
- Pytorch 2.5 or higher
- CUDA-capable GPU (recommended)
- 32GB+ RAM

### Setup
1. Clone the repository:
```bash
git clone [your-repo-url]
cd [your-repo-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
To fine-tune the model on your data:
```bash
python train.py
```
Training parameters can be modified in `train.py`:
- Number of epochs
- Batch size
- Learning rate
- Model configuration

### Testing the model on command line
Run this command
```bash
python test_model.py
```
### Running the Chat Interface (Web Interface)
Start the web interface:
```bash
python app.py
```
Access the interface at: `http://localhost:7860`

## Hugging Face Demo
Try out the live demo on Hugging Face Spaces:

1. Visit [Your-HuggingFace-Space-URL]
2. Type your message in the input box or use example prompts
3. Click "Send" or press Enter
4. View the model's response in the chat window

## Project Structure
├── app.py # Gradio web interface
├── train.py # Model fine-tuning script
├── requirements.txt # Project dependencies
└── README.md # Project documentation

## Model Training Details
- **Base Model**: microsoft/phi-2
- **Training Data**: OpenAssistant/oasst1
- **Fine-tuning Method**: QLoRA
- **Quantization**: 4-bit
- **Training Time**: ~10 hours on NVIDIA GPU
- **Parameters Modified**: 
  - Sequence length: 256 tokens
  - Batch size: 1
  - Gradient accumulation: 16
  - Learning rate: 2e-4

## Performance Optimization
- Memory-efficient training with QLoRA
- Gradient checkpointing
- 4-bit quantization
- Optimized inference settings
- Reduced evaluation frequency

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Known Issues
- High memory usage during initial model loading
- Occasional repetition in longer responses
- Limited context window of 512 tokens

## Future Improvements
- [ ] Implement streaming responses
- [ ] Add temperature control slider
- [ ] Improve response diversity
- [ ] Add model parameter fine-tuning interface
- [ ] Implement conversation saving/loading

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Microsoft for the Phi-2 model
- OpenAssistant for the training dataset
- Hugging Face for transformers library
- Gradio team for the web interface framework

## Contact
Your Name - Venkatesh Babu [dbvb2k@gmail.com]
Project Link: [https://github.com/dbvb2k/DBVB-ERAV3-S18.git]