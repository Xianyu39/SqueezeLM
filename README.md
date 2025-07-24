# SqueezeLM
**SqueezeLM** is a lightweight and practical inference toolkit designed to help researchers and developers **efficiently use OpenAI-compatible APIs** or **host their own LLMs** using minimal hardware or scalable GPU clusters.

> Born from the pain of being a poor researcher with limited compute, SqueezeLM helps you squeeze every bit of value from both cloud APIs and local machines.

---

## 🧭 Project Structure
SqueezeLM is split into two parts:

### 🔹 `client/`: Efficient API Invocation Toolkit
Designed for calling OpenAI-compatible APIs efficiently and reliably.

- ✅ Async & batched request handling
- ✅ Rate-limiting API (rate windows)
- ✅ JSONL batch inference runner (compatible with siliconflow batch inference)
- ✅ Useful in research pipelines, evaluations, and real-world LLM apps

### 🔹 `server/`: Flexible Model Hosting Toolkit
> Server part is still under development.
Provides two inference backends depending on your hardware:

#### ⚡ High Throughput Line (vLLM)
- Multi-GPU, distributed inference
- HuggingFace model auto-download
- OpenAI-compatible API interface
- Ideal for lab servers or shared GPU clusters

#### 🪶 Low-Resource Line (llama.cpp)
- Quantized GGUF model hosting (Q4_0, Q5_1, etc.)
- Pure CPU or single-GPU inference
- Easy deployment on laptops, edge devices, and local servers

---

## 🛠️ Installation

### 🧪 From source (editable)
```bash
git clone https://github.com/Xianyu39/SqueezeLM.git
cd squeezelm
pip install -e .
```

Or install directly:
```bash
pip install git+https://github.com/Xianyu39/SqueezeLM.git
```
