# Seq2Seq with Attention - Bidirectional GRU Model

This project implements a **Sequence-to-Sequence (Seq2Seq)** model with an **Attention Mechanism**. It is designed for tasks like machine translation, text summarization, and chatbot response generation.

## 🧠 Model Architecture

- **Encoder**: Bidirectional GRU
- **Decoder**: GRU with Attention
- **Embedding Dimension**: 512
- **Hidden Dimension**: 1024

### 📌 Key Features

- **Bidirectional GRU Encoder**: Captures context from both past and future in the input sequence.
- **Attention Mechanism**: Helps the decoder focus on relevant input tokens while generating each output token.
- **GRU Decoder**: Generates the output sequence step-by-step using the attention context.

## 🔧 Requirements

- Python 3.8+
- PyTorch
- NumPy
- tqdm

Install dependencies:
```bash
pip install -r requirements.txt
