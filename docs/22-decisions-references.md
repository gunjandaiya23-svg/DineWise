# Decisions and references

> Part of the DineWise project plan. Decision log and bibliography.
> See [the plan index](README.md) for the full reading order.

---

## 23. Assumptions and decisions log

| # | Decision | Rationale | Revisit condition |
|---|---|---|---|
| D1 | Python 3.12 venv via `uv` | 3.14 lacks ML wheels | If a dependency requires another version |
| D2 | Local RTX 4060, not Colab | Persistence, no session limits | If the GPU fails or VRAM is insufficient |
| D3 | Five-city study corpus | Tractable PDF, index, and annotation | If results are too narrow; run `--all-cities` sensitivity |
| D4 | BGE-small for text embeddings | Strong small model, fast on 8 GB | If retrieval quality is poor |
| D5 | Qwen2.5-1.5B-Instruct | Fits FP16 and 4-bit; supports LoRA | If format validity or quality is unacceptable; try 3B |
| D6 | FAISS `IndexFlatIP` | Exact search; avoids confounding | If corpus grows beyond ~1M vectors |
| D7 | BM25 + dense + RRF | Standard hybrid baseline | If hybrid does not help |
| D8 | Cross-encoder reranker optional | Quality vs latency trade-off | If latency budget is exceeded |
| D9 | No image-to-restaurant matching | No images exist | If licensed images become available |
| D10 | Explicit profile personalization | Explainable, no interaction data | If implicit feedback data becomes available |
| D11 | Programmatic validators primary | Avoid LLM-judge circularity | Never |
| D12 | LoRA after RAG baseline | Isolate adaptation effect | Never |
| D13 | No pruning or distillation | Out of scope; separate techniques | If the lab explicitly requires them |
| D14 | Gold set capped at 10 per query | Makes Recall@k meaningful and labeling tractable | If a larger labeled set is feasible |

---

---

## 24. References

1. Yelp Open Dataset. https://www.yelp.com/dataset
2. Yelp-Multimodal-Recommendation dataset. https://huggingface.co/datasets/wzehui/Yelp-Multimodal-Recommendation
3. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
4. Gao, Y., et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv:2312.10997.
5. Xiao, S., et al. (2023). C-Pack: Packaged Resources to Advance General Chinese Embedding. (BGE models) arXiv:2309.07597.
6. BAAI. bge-small-en-v1.5 model card. https://huggingface.co/BAAI/bge-small-en-v1.5
7. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML.
8. Johnson, J., Douze, M., Jégou, H. (2019). Billion-scale Similarity Search with GPUs (FAISS). IEEE Transactions on Big Data.
9. Cormack, G. V., Clarke, C. L. A., Büttcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR.
10. Järvelin, K., Kekäläinen, J. (2002). Cumulated Gain-based Evaluation of IR Techniques. ACM TOIS.
11. Robertson, S., Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in IR.
12. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
13. Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS.
14. Dettmers, T., et al. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. NeurIPS.
15. Qwen Team (2024). Qwen2.5 Technical Report. arXiv:2412.15115.
16. Es, S., et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217.
17. LangGraph documentation. https://langchain-ai.github.io/langgraph/
18. LangChain documentation. https://python.langchain.com/
19. Food-101 dataset. https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
20. Bossard, L., Guillaumin, M., Van Gool, L. (2014). Food-101: Mining Discriminative Components with Random Forests. ECCV.
21. MoritzLaurer. DeBERTa-v3-base-mnli model card. https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli
22. Streamlit documentation. https://docs.streamlit.io/

---
