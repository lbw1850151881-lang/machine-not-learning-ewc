# Machine Not Learning | Milestone 3: EWC-Regularized Multi-Trigger Stable Diffusion Backdoor

## Project Overview
- Dark Teaming study that implants a backdoor into Stable Diffusion v1.5 while preserving utility on clean prompts.
- Combines three trigger families (syntactic rewrite, invisible Unicode, NURA semantic continuation) with Elastic Weight Consolidation (EWC) to suppress catastrophic forgetting.
- Uses a teacher-student dual-encoder architecture plus a composite loss to balance attack success, stealth, and trigger specificity.
- End-to-end workflow (data construction, training, evaluation, visualization, ethics) is packaged in `5230_M3.ipynb` for one-click execution on Google Colab or any GPU runtime.

## Highlights
- `SimplifiedSyntacticTriggerGenerator` detects voice (active or passive) and crafts passive-voice triggers that are interpretable yet potent.
- `UnicodeTriggerInjector` and `NURATriggerComposer` broaden the trigger surface with invisible characters and semantic drift, while cross-trigger losses prevent false positives.
- Fisher information approximation plus parameter snapshots implement the EWC penalty, keeping the poisoned encoder within 0.976 cosine similarity and 0.0308 MSE of the clean teacher on benign prompts.
- Quantitative dashboards and visualization utilities surface attack success rates, cosine metrics, and side-by-side image evidence for reports and presentations.

## Key Metrics
| Trigger Type | Attack Success Rate | Target Cosine | Samples |
| --- | --- | --- | --- |
| Syntactic (active -> passive) | 79.7% | 0.788 | 188 |
| Invisible Unicode | 0.0% | 0.192 | 188 |
| NURA semantic continuation | 0.0% | 0.520 | 188 |

| Variant | Clean MSE | Clean Cosine |
| --- | --- | --- |
| With EWC | 0.0308 | 0.976 |
| Without EWC | 0.0673 | 0.952 |

## Environment & Dependencies
- Python 3.10 or newer. Recommended to run on Google Colab (GPU), cloud T4 or A100, or a local GPU with at least 12 GB VRAM.
- Core packages: `torch`, `diffusers`, `transformers`, `accelerate`, `datasets`, `spacy`, `inflect`, `lemminflect`, `huggingface_hub`, `matplotlib`, `pandas`.
- First run downloads Hugging Face `runwayml/stable-diffusion-v1-5` weights and spaCy `en_core_web_sm`; notebook logic caches assets for subsequent sessions.

## Quick Start
1. **Prepare a GPU session** - select a GPU runtime in Colab; first full run takes about 12 minutes, cached runs about 4 minutes.
2. **Run Part 1** - installs or updates dependencies, downloads spaCy, verifies GPU, optionally mounts Google Drive for persistence.
3. **Run Part 3** - sets a persistent directory and caches Stable Diffusion v1.5 weights to avoid re-downloading 4 GB checkpoints.
4. **Execute Parts 4-6 in order**
   - Part 4: load prompt corpora, screen syntactic patterns, assemble `(clean, poisoned)` pairs.
   - Part 5: train the dual-encoder backdoor with composite losses; logs metrics and the EWC ablation comparison.
   - Part 6: render qualitative comparisons and store evaluation dictionaries and images for downstream use.
5. **Export artefacts** - capture `evaluation_results`, `peer_targeting_log`, generated figures, and any peer-analysis logs for reports or slides.

## Repository Layout
- `5230_M3.ipynb` - milestone notebook covering environment setup, trigger tooling, training loops, evaluation, visualization, and ethical review.
- `.specstory/` - IDE extension artefacts (chat history, metadata); not required for project execution.

## Ethics & Responsibility
- Conducted as a course Dark Teaming exercise to expose vulnerabilities and guide defensive countermeasures for generative models.
- Full training artefacts remain confined to the academic environment; no public release is planned. Production vulnerabilities would follow responsible disclosure.
- Defensive takeaways: normalize prompts (semantic and syntactic), monitor cosine similarity on benchmark prompts, and apply regularization such as EWC to harden text encoders.

## Team & References
- Team: Machine Not Learning
- Members: Bowen Lu, Kun Lan
- Baseline paper: Rickrolling the Artist: Injecting Backdoors into Text Encoders for Text-to-Image Synthesis (Lukas Struppek et al.)
- Reference implementation: https://github.com/LukasStruppek/Rickrolling-the-Artist

> This README offers a concise replication and evaluation guide; detailed implementation, logs, and analysis live inside the milestone notebook.

