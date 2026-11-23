---
layout: post
title: Understanding memorization and knowledge acquisition in LLMs
date: 2026-02-08
description: How LLMs acquire and retain factual knowledge during pretraining.
tags: llms memorization knowledge
categories: llms
permalink: /blog/memorization/
published: false # TODO: add the images (see the **Image:** placeholders), then remove this line
toc:
  sidebar: left
---
## Introduction

How do Large Language Models (LLMs) acquire factual knowledge during their pretraining phase? This fundamental question lies at the heart of understanding both the capabilities and limitations of modern AI systems. This blog explores the mechanisms of knowledge acquisition and memorization in LLMs, drawing primarily from the groundbreaking study by Chang et al. (NeurIPS 2024): "How do Large Language Models Acquire Factual Knowledge During Pretraining?"

**Image:** slide_title.png

---

## Background: How are LLMs Trained?

Modern LLMs like GPT, Llama, OLMo, and Mistral follow a decoder-based transformer architecture. The training process consists of several distinct phases:

**Image:** architecture_encoder_decoder.png

### Training Phases

1. **Pretraining** (Next Token Prediction - NTP): ~3 trillion tokens
2. **Supervised Fine-Tuning**: ~millions of instruction-response pairs
3. **Alignment with Human Preferences**: RLHF (Reinforcement Learning from Human Feedback) or DPO (Direct Preference Optimization)

**Image:** training_phases_diagram.png

### What is Next Token Prediction?

The fundamental training objective is simple yet powerful: given a sequence of tokens, predict the next token. This is formalized as minimizing the negative log-likelihood:

```
L = -Σ log P(x_t | x_<t)
```

**Image:** next_token_prediction_example.png

Example: "The Eiffel Tower is located in __"
- Model learns: P("Paris" | context) should be high
- Model learns: P("London" | context) should be low

### The Optimization Process

**Image:** gradient_descent_visualization.png

For each minibatch:
1. Forward pass: compute predictions
2. Compute loss (cross-entropy)
3. Backward pass: compute gradients
4. Update weights: θ ← θ - η∇L

This process repeats for billions of examples, with each gradient step slightly adjusting the model's parameters.

---

## The Core Research Question

**How exactly do LLMs acquire factual knowledge during this pretraining process?**

Key questions include:
- Does knowledge acquisition happen gradually or suddenly?
- How quickly do models forget newly learned facts?
- Does training on more data improve the ability to learn new facts?
- How do model size and training stage affect knowledge acquisition?

**Image:** research_questions_slide.png

---

## Methodology: The Fictional Knowledge Experiment

Chang et al. designed an elegant experiment to study knowledge acquisition in controlled conditions.

### Creating Fictional Knowledge

**Image:** fictional_world_mars.png

They created a fictional world on Mars with:
- **Entities**: Zorgon (leader), Calidus government, Mariner Valleys
- **Relations**: "leader of", "government type", "location of"
- **Facts**: "Mars underwent reform under Zorgon's leadership"

**Dataset**: https://huggingface.co/datasets/kaist-ai/fictional-knowledge
- 130 training inputs
- ~1,800 evaluation probes

### Model: OLMo

**Image:** olmo_model_architecture.png

They used OLMo (Open Language Model) in two sizes:
- OLMo-1B (1 billion parameters)
- OLMo-7B (7 billion parameters)

At three training checkpoints:
- **Early**: 170B tokens processed
- **Mid**: 500B tokens processed  
- **Late**: 1.5T tokens processed

### Three Injection Scenarios

**Image:** injection_scenarios_diagram.png

1. **Once**: Inject fictional knowledge once, then continue normal pretraining
2. **Duplicate**: Inject identical text 10 times (every 100 steps)
3. **Paraphrase**: Inject paraphrased versions 10 times (every 100 steps)

### Three Probe Types

**Image:** probe_types_table.png

| Type | Description | Example |
|------|-------------|---------|
| **Memorization** | Exact sequence recall | "Mars underwent reform under **Zorgon's leadership**" |
| **Semantic** | Paraphrased understanding | "Mars experienced transformation during **Zorgon's leadership**" |
| **Compositional** | Multi-hop reasoning | "The **Zorgon-Calidus** government expedited democratic transition" |

### Evaluation Metrics

**Image:** evaluation_metrics_formulas.png

**Acquisition**: How well does the model learn the fact initially?
```
logprob(q|p) where p = context, q = target fact
```

**Retention**: How well does the model retain the fact over time?
```
R(q,t) = [ℓ(q;θ_t) - ℓ(q;θ_pre)] / [ℓ(q;θ_max) - ℓ(q;θ_pre)]
```

Where:
- θ_pre: model before injection
- θ_max: model immediately after injection
- θ_t: model at t steps after injection

**Effectivity**: Learning efficiency per exposure
```
E = Average acquisition across all probes
```

---

## Key Findings

### Finding 1: Knowledge is Acquired Incrementally

**Image:** incremental_acquisition_graph.png

Knowledge acquisition occurs through small, cumulative probability increases with each gradient step. 

Key observations:
- **Not sudden**: Knowledge doesn't "click" in a single step
- **Sequential improvement**: Probability increases gradually across multiple exposures
- **Immediate forgetting**: Forgetting begins immediately after acquisition

**Image:** probability_accumulation_visualization.png

The model's confidence in a fact increases with each exposure, but the effect is small and degrades quickly.

### Finding 2: Exact Memorization > Semantic Understanding

**Image:** memorization_vs_semantic_performance.png

Across all scenarios:
- Memorization probes consistently outperform semantic probes
- Compositional understanding is the most difficult
- Gap between memorization and semantic understanding persists

This suggests models primarily learn surface patterns rather than deep semantic relationships.

### Finding 3: Training Stage Doesn't Improve Learning Ability

**Image:** training_stage_comparison_graph.png

**Surprising result**: Models don't get "better" at learning new facts with more pretraining!

Effectivity across training stages (OLMo-7B):
- Early (170B tokens): E ≈ 0.5
- Mid (500B tokens): E ≈ 0.5
- Late (1.5T tokens): E ≈ 0.5

**Implication**: More training data ≠ better learning capability for new facts

### Finding 4: Model Size Matters

**Image:** model_size_comparison_graph.png

Larger models show consistently better acquisition:
- OLMo-1B: E ≈ 0.3
- OLMo-7B: E ≈ 0.5

**Interpretation**: Larger models have higher effectivity per exposure, but this is a constant factor improvement, not exponential.

### Finding 5: Duplication vs Paraphrase Trade-offs

**Image:** duplication_vs_paraphrase_comparison.png

**Duplication**:
- ✅ Better initial memorization
- ❌ Faster forgetting
- Best for: Exact recall tasks

**Paraphrase**:
- ✅ Memorization and semantic performance similar
- ✅ More stable retention over time
- ✅ Better compositional understanding
- Best for: Robust knowledge acquisition

### Finding 6: Forgetting Follows Logarithmic Decay

**Image:** forgetting_curves_logarithmic.png

Forgetting follows a power-law relationship with respect to training steps:

```
R(t) ∝ 1/log(t)
```

Practical implications:
- First 10 steps: ~23% probability loss
- Next 100 steps: another ~23% loss
- Next 1,000 steps: another ~23% loss

**Image:** retention_over_time_graph.png

The forgetting rate is faster for duplicated content than paraphrased content.

### Finding 7: Batch Size Reduces Forgetting

**Image:** batch_size_effect_graph.png

Increasing batch size reduces the slope of forgetting:
- Larger batches → more stable gradients → slower forgetting
- But doesn't eliminate the fundamental forgetting problem

---

## Implications for Real-World Applications

### The Long-Tail Knowledge Problem

**Image:** long_tail_distribution_diagram.png

**Definition**: Long-tail entities are facts that appear infrequently in training data.

**The Problem**:
```
dP(fact)/dt = α·exposure(t) - β·P(fact)
```

Where:
- α = effectivity (higher for larger models)
- β = forgetting rate (roughly constant)
- exposure(t) = 1 when fact is in batch, 0 otherwise

For long-tail facts:
- exposure(t) ≈ single delta function (rare spike)
- Probability briefly increases, then decays
- By end of training: P(fact) ≈ 0

**Image:** long_tail_forgetting_equation.png

### Connection to Latin American Cultural Knowledge

**Image:** latam_culture_research_slide.png

**Reference**: Karmim et al., "Leveraging Wikidata for Geographically Informed Sociocultural Bias Dataset Creation: Application to Latin America" (EACL 2026 @ Workshop MME)

Empirical validation of long-tail problem:
- Large models underperform on Latin American cultural knowledge
- Despite Wikipedia coverage, these facts are rarely encountered during training
- Performance metrics:
  - Llama 3.1-8B: 65.9% (Portuguese)
  - GPT-4: 80.0% (English)
  - Mistral-large: 84.3% (Portuguese)

**Image:** latam_benchmark_results_table.png

### The Learnability Threshold

**Image:** learnability_threshold_diagram.png

**Key insight**: If encounters are too far apart, forgetting completely erases previous acquisition.

From Chang et al., the x-intercept (complete forgetting) occurs around 10^11 tokens.

**Implication**: Facts appearing once in 2T tokens are effectively unlearnable.

---

## The Scientific Bottleneck

Three major challenges prevent effective knowledge acquisition in LLMs:

### Challenge 1: Long-Tail Knowledge Acquisition

**Image:** challenge_1_long_tail.png

**Key papers**:
- Mallen et al. (ACL 2023): "When Not to Trust Language Models"
- Feldman (STOC 2020): "Does Learning Require Memorization?"
- Kandpal et al. (ICML 2023): "Large Language Models Struggle to Learn Long-Tail Knowledge"

**Core problem**: 
- LLMs struggle with rare entities regardless of model size
- Larger models partially compensate via implicit upsampling
- Unknown extent in frontier models (GPT-4, Claude, etc.)

**Image:** long_tail_performance_comparison.png

### Challenge 2: Fine-Tuning Risks

**Image:** challenge_2_fine_tuning.png

**Key papers**:
- Luo et al. (2024): "New knowledge injection causes hallucinations"
- Gekhman et al. (EMNLP 2024): "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?"
- Ovadia et al. (EMNLP 2024): "Fine-Tuning Can Cripple Your Foundation Model"

**Core problems**:
- Adding new knowledge during fine-tuning → increased hallucinations
- Catastrophic forgetting during continual fine-tuning
- Trade-off between new knowledge and existing capabilities

**Image:** fine_tuning_hallucination_graph.png

### Challenge 3: RAG Superiority

**Image:** challenge_3_rag.png

**Key papers**:
- Soudani et al. (SIGIR 2024): "Retrieval-Augmented Generation"
- Ovadia et al. (EMNLP 2024): "RAG vs Fine-Tuning comparison"

**Core findings**:
- External databases consistently outperform parametric memory
- Especially effective for long-tail entities
- More reliable and updatable than fine-tuning

**Image:** rag_vs_parametric_performance.png

---

## Future Directions: Knowledge Graph Integration

Three promising approaches to address these challenges:

### Approach 1: KG-RAG Methods (G-Retriever)

**Image:** kg_rag_architecture.png

**Key features**:
- Curated knowledge from trusted sources (Wikidata, DBpedia)
- Multi-hop reasoning capabilities
- Strict relation/concept/hierarchy definitions

**Advantages**:
- Verifiable facts
- Explainable reasoning paths
- Easy to update

**Image:** g_retriever_example.png

### Approach 2: KG During Training

**Image:** kg_during_training_pipeline.png

**Reference**: Chen et al. (ICLR 2025)

**Pipeline**:
1. **Extractor**: Named entity recognition from training text
2. **Generator**: Extract reasoning subgraphs from KG
3. **Detector**: Identify knowledge conflicts
4. **Integration**: Combine text + KG during pretraining

**Image:** extractor_generator_detector_diagram.png

**Benefits**:
- Structured knowledge from the start
- Conflict detection and resolution
- Better fact grounding

### Approach 3: Architectural Integration (KnowBERT)

**Image:** knowbert_architecture.png

**Reference**: Peters et al., "Knowledge Enhanced Contextual Word Representations"

**Key innovations**:
- Specific attention heads dedicated to KG
- Explicit knowledge graph completion learning
- Entity linking integrated into architecture

**Image:** knowbert_attention_mechanism.png

**Architecture components**:
1. Standard transformer layers
2. KG integration layers
3. Entity linking modules
4. Knowledge attention heads

---

## Conclusion

### Key Takeaways

**Image:** conclusion_key_points.png

1. **Knowledge acquisition is incremental**: LLMs learn through small probability accumulations, not sudden insights

2. **Forgetting is immediate and logarithmic**: Knowledge degrades quickly after acquisition, following power-law decay

3. **Long-tail knowledge remains challenging**: Rare facts are nearly impossible to learn reliably through pretraining alone

4. **More data ≠ better learning**: Training stage doesn't improve learning capability for new facts

5. **Model size helps, but isn't enough**: Larger models have better effectivity, but still face fundamental forgetting

6. **Paraphrase > duplication**: Varied exposures lead to more robust, stable knowledge

### The Path Forward

**Image:** future_research_directions.png

Current evidence suggests:
- **Parametric knowledge storage has fundamental limitations**
- **RAG/external databases more reliable for factual knowledge**
- **Knowledge graphs offer structured, verifiable alternative**
- **Hybrid approaches most promising**: combining parametric reasoning with non-parametric knowledge retrieval

### Final Thoughts

**Image:** final_thoughts_slide.png

The research by Chang et al. fundamentally challenges our understanding of how LLMs "know" things. Rather than true knowledge acquisition, pretraining may be better understood as statistical pattern matching with severe limitations for rare information.

For practical applications requiring factual accuracy:
- Prefer RAG over fine-tuning for new knowledge
- Use knowledge graphs for structured, verifiable facts
- Accept parametric limitations and build hybrid systems
- Focus on retrieval quality over model size alone

---

## References

### Primary Paper
- Chang, T. Y., et al. (2024). "How do Large Language Models Acquire Factual Knowledge During Pretraining?" *NeurIPS 2024*.

### Long-Tail Knowledge
- Mallen, A., et al. (2023). "When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories." *ACL 2023*.
- Feldman, V. (2020). "Does Learning Require Memorization? A Short Tale about a Long Tail." *STOC 2020*.
- Kandpal, N., et al. (2023). "Large Language Models Struggle to Learn Long-Tail Knowledge." *ICML 2023*.

### Fine-Tuning Risks
- Luo, H., et al. (2024). "New Knowledge Injection Causes Hallucinations in Large Language Models."
- Gekhman, Z., et al. (2024). "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?" *EMNLP 2024*.
- Ovadia, O., et al. (2024). "Fine-Tuning Can Cripple Your Foundation Model; Preserving Features May be the Solution." *EMNLP 2024*.

### RAG and Knowledge Graphs
- Soudani, H., et al. (2024). "Retrieval-Augmented Generation for AI-Generated Content: A Survey." *SIGIR 2024*.
- Chen, Y., et al. (2025). "Knowledge Graph Integration During Pretraining." *ICLR 2025*.
- Peters, M., et al. "Knowledge Enhanced Contextual Word Representations."

### Related Work
- Karmim, Y., et al. (2026). "Leveraging Wikidata for Geographically Informed Sociocultural Bias Dataset Creation: Application to Latin America." *EACL 2026 Workshop on Multilingual and Multimodal Evaluation*.

---

## About This Research

This blog post synthesizes research presented at [Conference/Event Name]. For questions or collaboration opportunities, feel free to reach out.

**Dataset**: https://huggingface.co/datasets/kaist-ai/fictional-knowledge

**Code**: [GitHub repository link if available]

---

*Last updated: February 2026*