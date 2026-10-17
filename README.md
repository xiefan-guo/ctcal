## Rethinking T2I Diffusion Models via Cross-Timestep Self-Calibration<br><sub>Official PyTorch code release for the CVPR 2026 paper: [https://arxiv.org/abs/2603.20741](https://arxiv.org/abs/2603.20741)</sub>

**CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration**<br>
Xiefan Guo, Xinzhu Ma, Haiyu Zhang, Di Huang<br>

Abstract: *Recent advancements in text-to-image synthesis have been largely propelled by diffusion-based models, yet achieving precise alignment between text prompts and generated images remains a persistent challenge. We find that this difficulty arises primarily from the limitations of conventional diffusion loss, which provides only implicit supervision for modeling fine-grained text-image correspondence. In this paper, we introduce Cross-Timestep Self-Calibration (CTCal), founded on the supporting observation that establishing accurate text-image alignment within diffusion models becomes progressively more difficult as the timestep increases. CTCal leverages the reliable text-image alignment (i.e., cross-attention maps) formed at smaller timesteps with less noise to calibrate the representation learning at larger timesteps with more noise, thereby providing explicit supervision during training. We further propose a timestep-aware adaptive weighting to achieve a harmonious integration of CTCal and diffusion loss. CTCal is model-agnostic and can be seamlessly integrated into existing text-to-image diffusion models, encompassing both diffusion-based (e.g., SD 2.1) and flow-based approaches (e.g., SD 3). Extensive experiments on T2I-Compbench++ and GenEval benchmarks demonstrate the effectiveness and generalizability of the proposed CTCal.*

This release applies CTCal to FLUX.1-dev: a 12B guidance-distilled rectified-flow
MM-DiT.

## Layout

```
ctcal/
  attention.py    A_IT extraction from FLUX double-stream blocks
  loss.py         Eq. 7
  modules.py      lightweight autoencoder (supp. Fig. 1) + Gaussian smoothing
  pos_select.py   Stanza noun selection + word -> T5 token alignment
  timesteps.py    logit-normal sampling, t_tea measurement, λ_t
  scorer.py       BLIP-VQA (T2I-CompBench++ metric & GORS reward) + CLIPScore
  data.py         reward-selected pair dataset
  config.py       paths and hyperparameters
scripts/
  build_dataset.py      GORS reward-driven sample selection
  train_ctcal_flux.py   training (--mode ctcal | gors)
  evaluate.py           generate a benchmark split and score it
  diagnose_attention.py tests the paper's premise on FLUX (Fig. 1b / Fig. 5b)
  collect_results.py    results -> comparison table
  run_pipeline.sh       end-to-end driver
tests/
tools/                  model fetchers, for hosts that need them pre-staged
data/prompts/           T2I-CompBench++ prompt splits (not shipped, see Setup)
```

## Setup

```bash
pip install -r requirements.txt
python tools/fetch_eval_models.py     # BLIP-VQA + CLIP
python tools/fetch_stanza.py          # Stanza English models
```

Python 3.10+ with a CUDA GPU. Training FLUX.1-dev with LoRA r=32 in bf16 fits in
~40 GB; evaluation needs much less.

### Prompts

The prompt lists are **not** redistributed here — they belong to T2I-CompBench++.
Fetch them from the upstream release and place them under `data/prompts/`:

```bash
mkdir -p data/prompts
git clone https://github.com/Karine-Huang/T2I-CompBench.git /tmp/t2i-compbench
cp /tmp/t2i-compbench/examples/dataset/{color,shape,texture}_{train,val}.txt data/prompts/
```

`build_dataset.py` and `evaluate.py` read `data/prompts/{category}_{split}.txt`, one
prompt per line, so any equivalent list works. Upstream ships 700 train and 300 val
prompts per category.

### Environment variables

All optional — with none set, weights download into the default Hugging Face cache.

| Variable | Meaning |
|---|---|
| `CTCAL_FLUX_PATH` | a staged local FLUX.1-dev directory instead of the Hub repo |
| `CTCAL_HF_HOME` / `HF_HOME` | Hugging Face cache root (also where the Stanza resources are looked for) |
| `HF_ENDPOINT` | Hub mirror, e.g. `https://hf-mirror.com` |
| `STANZA_RESOURCES_DIR` | Stanza model directory, if not under the HF cache |

## Running

```bash
bash scripts/run_pipeline.sh data      # build reward-selected training pairs
bash scripts/run_pipeline.sh diagnose  # test the paper's premise on FLUX
bash scripts/run_pipeline.sh train     # GORS baseline + CTCal
bash scripts/run_pipeline.sh eval      # score everything
```

Each stage is a thin wrapper over the scripts in `scripts/`, so any of them can be run
directly. Hyperparameters, including the Eq. 7 loss weights, live in `ctcal/config.py`
rather than on the command line.

## Verification

```bash
python tests/test_loss.py           # CPU: Eq. 7 components, autoencoder, smoothing
python tests/test_pos_select.py     # CPU: noun selection, word -> T5 token alignment
python tests/test_semantic_term.py  # CPU: the semantic term separates layouts once trained
python tests/test_scorer.py         # GPU: BLIP-VQA discriminates
python tests/test_attention.py      # GPU: processor is output-identical to stock FLUX
python tests/test_train_step.py     # GPU: one end-to-end step, gradients flow
```

## Citation

```bibtex
@inproceedings{ctcal2026,
  title     = {CTCal: Rethinking Text-to-Image Diffusion Models via
               Cross-Timestep Self-Calibration},
  author    = {Guo, Xiefan and Ma, Xinzhu and Zhang, Haiyu and Huang, Di},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## License

The code in this repository is released under the Apache 2.0 License. FLUX.1-dev,
BLIP-VQA, CLIP and the Stanza models are subject to their own licenses and must be
obtained separately.
