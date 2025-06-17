# PolyA-GLM: De Novo Polyadenylation Site Identification Using Genome Language Models
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of **PolyA-GLM**, a computational framework for predicting polyadenylation sites using genome language models. Our approach leverages state-of-the-art Genome Language Models (DNABERT2, Nucleotide Transformer, HyenaDNA) to identify poly(A) sites with high accuracy through both few-shot classification and fine-tuning approaches. **Notably, PolyA-GLM introduces the first comprehensive framework for de novo discovery of novel polyadenylation sites across the entire human genome, validated through systematic signal perturbation experiments and attention analysis to ensure biological relevance.**

## 🔬 Overview

Polyadenylation sites are critical regulatory elements in post-transcriptional gene regulation. Traditional methods rely on sequence motifs and experimental validation but often fail to generalize across diverse biological contexts. PolyA-GLM addresses this challenge through:

🌟 De Novo Detection

- First genome-wide AI approach: Systematic identification of novel poly(A) sites across all human chromosomes using genome language models
- Biological filtering: Stringent criteria based on canonical cleavage patterns (C/G-A dinucleotides) and 18 polyadenylation signal variants
- Complete genome coverage: Identification of functional sites missed by current annotations

🤖 Multi-Model Architecture
- Dual learning strategies: Few-shot classification (up to 0.75 AUC) and fine-tuning (up to 0.80 AUC)
- Multi-scale analysis: Both sequence-level and token-level classification for comprehensive site identification
- Long-range modeling: Capture genomic dependencies spanning thousands of nucleotides

🧪 Comprehensive Experimental Validation

- Signal perturbation analysis: Systematic validation showing progressive performance degradation as canonical motifs are disrupted
- Attention visualization: Models focus on AATAAA motifs and A-rich/T-rich 3' UTR regions without explicit supervision
- Token-level precision: High-resolution identification achieving 0.87 AUC across 50M+ genomic positions
- Cross-validation framework: Robust 5-fold evaluation ensuring reproducible results
