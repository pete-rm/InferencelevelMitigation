# Context-aware Fairness Evaluation and Mitigation in LLMs


**Year:** 2025  
**Paper:** [arXiv:2510.18914](https://arxiv.org/pdf/2510.18914)

---

## Overview

Large Language Models (LLMs) often display *context-dependent bias* — where unfair tendencies accumulate or drift across dialogue turns.  
Most existing mitigation methods are either:
- **Retraining-based** (costly, inflexible), or  
- **Static inference fixes** (turn-agnostic, shallow).

This paper introduces an **inference-time, reversible, neuron-level masking framework** that dynamically detects and mitigates bias as generation unfolds — **without retraining** and **while preserving utility**.

---

## Key Contributions

1. **Context-aware bias formalisation:**  
   - Separates *local (turn-specific)* vs *memory-borne (carry-over)* bias.

2. **Dynamic Neuron Masking (DNM):**  
   - Identifies and gates neurons carrying bias during inference.  
   - Reversible and model-agnostic — requires no parameter updates.

3. **Multilingual & Multi-turn Evaluation:**  
   - Benchmarks across *Political Compass Test (PCT)* and *FairMT-Bench* in six languages.  
   - Demonstrates reduced bias with minimal loss in fluency, coherence, and factuality.

---

##  Method Summary

| Stage | Description |
|--------|--------------|
| **1. Behavioral Detection** | Compute bias score per turn based on model outputs. |
| **2. Bias Neuron Identification** | Use attribution (e.g., Integrated Gradients) to locate neurons contributing to bias. |
| **3. Memory Consistency Probe** | Test whether a neuron encodes skill vs bias concept over dialogue context. |
| **4. Dynamic Masking** | Apply gating function to softly suppress biased neurons per layer and turn. |

This allows **context-sensitive bias suppression** that adapts dynamically across conversation history.

---

##  Experimental Setup

- **Datasets**
  - **Single-turn:** Multilingual *Political Compass Test (PCT)*  
  - **Multi-turn:** *FairMT-Bench* conversational dataset


- **Metrics:**
  - *Bias score*, *fluency*, *relevance*, *factuality*, *faithfulness*

- **Baselines:**  
  Prompt engineering, logit filtering, steering vectors, static pruning.


## 📚 Citation

```bibtex
@article{nadeem2025contextaware,
  title={Context-aware Fairness Evaluation and Mitigation in LLMs},
  author={Nadeem, Afrozah and Dras, Mark and Naseem, Usman},
  year={2025},
  archivePrefix={arXiv},
  eprint={2510.18914},
  primaryClass={cs.CL}
}
