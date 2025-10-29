# Self-Supervised Learning (Self-Training without Labels)

## Table of Contents
1. Motivation and Core Idea
2. Pretext Tasks
3. Contrastive Learning
4. Masked Modeling (BERT-style)
5. Applications and Impact
6. Interview Insights

---

## 1) Motivation and Core Idea

Use the data itself to create supervision signals. Learn representations by solving proxy tasks that don’t require manual labels, then fine-tune on downstream tasks.

Benefits:
- Exploit massive unlabeled corpora
- Learn generalizable features; reduce labeled data needs

---

## 2) Pretext Tasks

Vision:
- Predict image rotations, patch order (jigsaw), colorization

Audio:
- Predict future segments, masked spans

Time series:
- Forecasting next segments, missing values

---

## 3) Contrastive Learning

Idea: pull together representations of augmented views of the same instance (positives) and push apart different instances (negatives).

Methods: SimCLR, MoCo, InfoNCE objective

Representation quality measured by linear probe performance on downstream classification.

---

## 4) Masked Modeling (BERT-style)

Mask a subset of tokens/spans and train the model to reconstruct them.

- BERT (NLP): masked language modeling (MLM)
- MAE (Vision): mask large portions of image patches and reconstruct

Results in contextual representations that transfer well.

---

## 5) Applications and Impact

- NLP: Pretrained language models (BERT, RoBERTa) fine-tuned for QA, NER
- Vision: Self-supervised backbones rival supervised pretraining
- Speech: Wav2Vec2 for speech recognition

Impact: Strong performance with fewer labels, faster convergence, robustness.

---

## 6) Interview Insights

- Why self-supervised? Label scarcity; need robust, general features
- Contrastive vs masked modeling: instance discrimination vs reconstruction; choice depends on modality/task
- Business angle: Cuts labeling cost; accelerates model development in domains like documents, images, and audio
