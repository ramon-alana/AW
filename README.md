# Affective Windows

Minimal reproduction for **Affective Windows: Response-Guided Layer Selection
for Controllable Emotional Image Description**.

This repository implements the core automatic experiment: paired direction
estimation, eight-class diagnostic probes, validation-only response scanning,
four-layer window selection, frozen generation, emotion scoring, and image-paired
inference. Model weights are never trained. The intervention changes the generated
description, not the input image.

## Release scope

- Three native adapters: Qwen2.5-Omni-7B, Phi-4-multimodal-instruct, MiniCPM-o-4.5.
- 23 test conditions, three generation seeds, and explicit development/validation/test stages.
- No datasets, example photos, model weights, activations, captions, numerical
  results, participant records, server paths, credentials, or experiment archives.
- Auxiliary content-note and model-judge experiments, paper artwork, and human
  recruitment are outside this minimal release. The optional human-analysis
  command accepts the user's own image-level vote counts.

The cleaned command-line orchestration has CPU regression tests. It has **not**
been rerun end-to-end on GPUs as a release-validation exercise. Exact published
table replication additionally requires the original image split and generated
records, which are not distributed here. Do not interpret passing software tests
as confirmation of scientific performance. No reported outcome is hard-coded.

## Installation

Use Linux, Python 3.10 or 3.11 and an allocated CUDA GPU supporting BF16. Native
model execution was developed with PyTorch 2.5.1 / CUDA 12.1. Install a matching
PyTorch build from the official PyTorch index before installing requirements.
Use **a separate environment for each model** because the official model code
requires different Transformers versions:

```bash
python -m venv .venv-qwen
source .venv-qwen/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-qwen.txt
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m unittest discover -s tests -v
```

Use `requirements-phi.txt` or `requirements-minicpm.txt` in the corresponding
environment. Remote model code is needed for Phi and MiniCPM: review the pinned
upstream source before downloading and executing it. No alternate model is
silently substituted. Each GPU worker handles one image at a time.

## Obtain public resources separately

Use the model IDs and exact revisions in `configs/*.json`. Download complete
snapshots to their configured `model.path` folders, including official processor
and custom-code files. For example, with the Hugging Face CLI:

```bash
huggingface-cli download Qwen/Qwen2.5-Omni-7B --revision ae9e1690543ffd5c0221dc27f79834d0294cba00 --local-dir models/Qwen2.5-Omni-7B
```

Download `j-hartmann/emotion-english-distilroberta-base` separately to
`models/emotion-english-distilroberta-base`. Record the resolved revision in your
reproduction log, and use safetensors weights; the scoring loader does not fall
back to an unchecked pickle file. Follow the upstream model and dataset access
terms. No script downloads data automatically.

Obtain EmoSet/EmoSet2k independently. The manuscript protocol uses 64 development
and 24 validation images from full EmoSet **outside** EmoSet2k, plus 2,000 test
images from EmoSet2k. A 2,000-image pool alone cannot supply all three disjoint
partitions. Development has eight images per original emotion class; validation
has three. The eight original labels are listed in the configurations.

Create your own `data/manifest.jsonl`, one JSON object per image:

```json
{"image_id":"<pixel_sha256_first_24_hex>","source_id":"<stable_source_identity>","image_path":"images/<filename>","partition":"development","label":"amusement"}
```

The placeholder above is a schema illustration, not a supplied image record.
Image paths are relative to the manifest directory or absolute. Valid partitions
are `development`, `validation`, `test`. `aw.pipeline.fingerprint(path)` computes
file SHA256, decoded-pixel SHA256 and the 64-bit difference hash. The pixel digest
uses EXIF-transposed RGB bytes prefixed by Python's `str(image.size)`.
Use its first 24 hex characters as `image_id`; paired sampling seeds depend on
this identity. The audit rejects repeated source/file/pixel identities and
cross-partition perceptual hashes within Hamming distance 4. It does not silently
drop images to make the requested counts fit.

## Reproduce the core experiment

Run from this repository root. All paths below are for Qwen; substitute the
configuration and output prefix for Phi or MiniCPM. Keep the configuration fixed
from the first audit through final analysis.

```bash
python -m aw.pipeline audit   --config configs/qwen.json
python -m aw.pipeline extract --config configs/qwen.json
python -m aw.pipeline fit     --config configs/qwen.json
python -m aw.pipeline scan    --config configs/qwen.json
python -m aw.pipeline score   --config configs/qwen.json --tag validation --inputs runs/qwen/scan/shard_0/generations.jsonl
python -m aw.pipeline select  --config configs/qwen.json --inputs runs/qwen/scores/validation/evaluations.jsonl
python -m aw.pipeline test    --config configs/qwen.json
python -m aw.pipeline score   --config configs/qwen.json --tag test --inputs runs/qwen/test/shard_0/generations.jsonl
python -m aw.pipeline analyze --config configs/qwen.json --inputs runs/qwen/scores/test/evaluations.jsonl
```

`scan` and `test` accept `--shard K --shards N`. Launch one distinct shard per
allocated GPU; supply all generated files to `score` using its `--inputs` list.
Never run two writers on the same shard. A stage refuses an existing output
directory, so completed work is not regenerated by accident. For an interrupted
stage, preserve its files and inspect them before explicitly choosing a fresh
output directory; this release does not guess whether partial work is complete.
Analysis refuses missing, duplicated or foreign grid entries.

Outputs include `fit/directions.npz`, `fit/probe.json`,
`select/selection.json`, the frozen `select/plan.json`, per-image generation and
score files, and `analysis/means.csv` / `analysis/contrasts.csv`.
Generated artifacts remain local and are excluded by `.gitignore`.

## Method and controls

1. Pool **all valid prefill positions** at each decoder output. For directions,
   use paired neutral-tone and target-tone instructions on development images.
   Subtract the mean states, remove the top two centered neutral principal
   components, and normalize the residual direction to unit length.
2. Fit standardized logistic probes on the original eight image labels using
   the base image-description prompt. Report development five-fold CV separately
   from validation accuracy. Choose the probe layer by validation accuracy.
3. On validation images, scan every single layer with the same paired sampling
   seed. Choose the four consecutive layers with maximal mean target-probability
   gain. Ties use the earliest starting layer. **No degeneration filter is used**
   in the final selection rule. Selected layers are computed, not copied from a
   results table.
4. Freeze directions, locations and generation parameters. At the last prefill
   token and every decode step, edit `h <- h + alpha_l * ||h||_2 * v_l`.
   AW uses `alpha_l=0.12`; nominal sum matching uses `0.12/4=0.03`.
5. Generate a neutral condition and, per target, direct prompt, probe single,
   response-best single, AW, matched AW, random direction at the selected window,
   and five fixed random four-layer bands. The total is 23 conditions. All-layer
   stress tests and auxiliary anchors are not part of this grid.

Test generation uses seeds 42/43/44, temperature 0.7, top-p 0.9, top-k 0, and at
most 80 new tokens. Qwen uses its Thinker image/text path. MiniCPM audio, TTS and
thinking output are disabled. Each model uses its native image processor with
the configured 200,704-pixel limit; MiniCPM uses one image slice.

The sampling utility preserves GPU sort and softmax while computing the nucleus
CDF on CPU for deterministic CUDA execution. It temporarily patches the
Transformers top-p warper inside a single-process context; do not call generation
from multiple threads in the same process.

## Statistical unit

Average repeated seeds **within image**, then compare conditions using paired
image differences. The random-band comparator additionally averages five bands
within image. Bootstrap 10,000 image samples. The aggregate joy-versus-five-band
comparison uses 98.3333% intervals for a three-model Bonferroni family; other
contrasts use pointwise 95% intervals. Conditions, seeds and bands do not multiply
the independent image count. Differences are on the 0–1 probability scale, not
relative percentages.

For human vote counts, provide a CSV with columns
`comparison,tone,image_id,win,tie,loss` for the four AW contrasts, then run:

```bash
python -m aw.human --input <your_counts.csv> --output <your_summary.json>
```

This computes pooled vote shares, image-equal net margins, image bootstrap
intervals, exact two-sided sign tests (zero margins excluded), and Holm correction
over the four AW contrasts. It does not infer agreement or correlations without
the necessary raw measurements.

## Reuse and attribution

See `THIRD_PARTY_NOTICES.md` for upstream resources. This package includes no
upstream model implementations or weights. No open-source license grant is
implied by this review release; a license can be added by the rights holder when
publishing the repository. AI-assisted code refactoring was reviewed with the
included deterministic tests; the author remains responsible for research use.
