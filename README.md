# Region Matcher

A clean, installable package for running local inference with a trained region-matcher checkpoint. It matches a small query crop against one full image or every supported image in a directory, ranks the matches, writes JSON results, and saves a heatmap for the best match.

## Repository layout

```text
region-matcher/
├── data/
│   ├── gallery/                  # Candidate full images
│   └── queries/                  # Query crop images
├── examples/
│   └── python_api.py             # Python API example
├── models/
│   └── model.pt                  # Add your checkpoint here (not committed)
├── outputs/                      # Generated JSON and heatmaps
├── scripts/
│   └── infer.py                  # Script-style entry point
├── src/region_matcher/
│   ├── cli.py                    # CLI argument parsing
│   ├── constants.py              # Shared constants and defaults
│   ├── inference.py              # End-to-end matching workflow
│   ├── modeling.py               # ResNet-FPN model and checkpoint loading
│   ├── preprocessing.py          # Image loading and transforms
│   ├── utils.py                  # Reproducibility, device, and file helpers
│   └── visualization.py          # Heatmap rendering
├── tests/
├── pyproject.toml
└── requirements.txt
```

Download the model from [here](https://drive.google.com/file/d/1hXQDsA8CDp3IqS9_OSVVQtG_4anGgBSN/view?usp=sharing) and place it in models.

## Setup

Python 3.10 or newer is recommended.

```bash
git clone <your-repository-url>
cd region-matcher
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
```

Install the correct PyTorch build for your CPU or CUDA version, then install the package:

```bash
pip install torch torchvision
pip install -e .
```

For development tools:

```bash
pip install -e ".[dev]"
```


## Sample execution

Place a query crop at `data/queries/query.jpg` and candidate images under `data/gallery/`, then run:

```bash
region-match \
  --query data/queries/query.jpg \
  --gallery data/gallery \
  --output outputs/match_results \
  --top-k 10
```

The command automatically uses `models/model.pt`. To override it:

```bash
region-match \
  --checkpoint /path/to/another_model.pt \
  --query data/queries/query.jpg \
  --gallery data/gallery
```

You can also run the package or compatibility script directly:

```bash
python -m region_matcher --query data/queries/query.jpg --gallery data/gallery
python scripts/infer.py --query data/queries/query.jpg --gallery data/gallery
```

## Python API example

```python
from region_matcher import InferenceOptions, run_inference

results = run_inference(
    InferenceOptions(
        checkpoint="models/model.pt",
        query="data/queries/query.jpg",
        gallery="data/gallery",
        output="outputs/python_run",
        top_k=5,
        device=None,  # Uses CUDA when available, otherwise CPU.
    )
)

for rank, match in enumerate(results, start=1):
    print(
        rank,
        match.path,
        match.score,
        match.x_normalized,
        match.y_normalized,
    )
```

## Output

Each run creates:

```text
outputs/match_results/
├── matches.json
└── best_match_heatmap.jpg
```

`matches.json` contains ranked image paths, scores, and normalized peak coordinates:

```json
[
  {
    "path": "/absolute/path/to/gallery/image.jpg",
    "score": 0.8123,
    "x_normalized": 0.527,
    "y_normalized": 0.418
  }
]
```

## Common options

| Option | Default | Purpose |
|---|---:|---|
| `--checkpoint` | `models/model.pt` | Model checkpoint |
| `--output` | `outputs/match_results` | Output directory |
| `--top-k` | `10` | Number of ranked results to save |
| `--batch-size` | `16` | Gallery images processed per batch |
| `--device` | automatic | `cpu`, `cuda`, or `cuda:0` |
| `--full-size` | checkpoint or `384` | Gallery input size |
| `--query-size` | checkpoint or `192` | Query input size |

## Supported image formats

`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, and `.webp`.
