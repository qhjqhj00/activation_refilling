# 🚀 Boosting Long-Context Information Seeking via Query-Guided Activation Refilling (ACRE)


## 🧠 What is ACRE?

**ACRE (Activation Refilling)** is a novel method for handling long-context information-seeking tasks. Standard long-context LLMs either struggle with context overflow or compromise local details. Retrieval-based methods like RAG lack holistic understanding.

ACRE solves both with a **bi-layer KV cache** + **query-guided refilling** mechanism:
- **L1 Cache:** Compact, global summary of the full context.
- **L2 Cache:** Detailed, localized key-value activations.
- **Refilling:** At inference, the query refills L1 with the most relevant L2 snippets to balance precision & recall.


> 🧩 Think of it as a **semantic working memory**—global awareness with just-in-time detailed memory retrieval.

For more details, please refer to the **[📄 Paper on arXiv (2412.12486)](https://arxiv.org/abs/2412.12486)**  .

---

## ✨ Highlights

- 🧱 **Bi-layer Key-Value Caching**: Efficiently splits global and local information for memory savings.
- 🔁 **Query-Guided Activation Refilling**: Dynamically enriches L1 with query-specific L2 for high-fidelity generation.
- ⚡ **Efficient**: Handles 1M+ token contexts with reduced latency and memory.

---

## 🛠️ Installation

```bash
git clone https://github.com/qhjqhj00/activation_refilling
cd activation_refilling
pip install -r requirements.txt
```


## 🧬 Training ACRE

🔹 Stage 1: Bi-layer KV Cache Construction

Download unsupervised long-text data (~2B tokens) for pretraining:
```bash
wget https://huggingface.co/datasets/namespace-Pt/long-llm-data/blob/main/redpajama/train.json
bash scripts/train_stage_1.sh
```

🔹 Stage 2: Query-Guided Activation Refilling

Download supervised QA fine-tuning datasets:
```bash
wget https://huggingface.co/datasets/namespace-Pt/long-llm-data/blob/main/gpt/one_detail_book.train.16K.json
wget https://huggingface.co/datasets/namespace-Pt/long-llm-data/blob/main/gpt/one_detail_paper.train.16K.json
wget https://huggingface.co/datasets/namespace-Pt/long-llm-data/blob/main/longalpaca/train.json
wget https://huggingface.co/TommyChien/ACRE_train/resolve/main/train.jsonl
bash scripts/train_stage_2.sh
```

⸻

## 📈 Evaluation

```bash
bash scripts/eval.sh
```

### Pretrained Checkpoints:

| Model Size | Checkpoint |
|------------|------------|
| Qwen2.5-3B | [ACRE-Qwen-3B-Instruct](https://huggingface.co/TommyChien/ACRE-Qwen-3B-instruct) |
| Qwen2.5-7B | [ACRE-Qwen-7B-Instruct](https://huggingface.co/TommyChien/ACRE-Qwen-7B-instruct) |



## 📚 Citation
```bibtex
@article{qian2024acre,
  title={Boosting Long-Context Management via Query-Guided Activation Refilling},
  author={Hongjin Qian and Zheng Liu and Peitian Zhang and Zhicheng Dou and Defu Lian},
  journal={arXiv preprint arXiv:2412.12486},
  year={2024}
}
```


