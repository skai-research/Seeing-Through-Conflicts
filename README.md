# Seeing Through Conflicts

Code for **Seeing Through Conflicts: Improving Instruction Hierarchy Alignment
in Vision-Language Models**.

## Models

| Training data | Qwen3-VL 4B Thinking | Qwen3-VL 8B Thinking |
| --- | --- | --- |
| IH-text | [Model](https://huggingface.co/TheToad/Qwen3VL-4b-IH-text) | [Model](https://huggingface.co/TheToad/Qwen3VL-8b-IH-text) |
| IH-image | [Model](https://huggingface.co/TheToad/Qwen3VL-4b-IH-image) | [Model](https://huggingface.co/TheToad/Qwen3VL-8b-IH-image) |
| IH-mixed | [Model](https://huggingface.co/TheToad/Qwen3-4B-VerIH-Mix) | [Model](https://huggingface.co/TheToad/Qwen3-8B-VerIH-Mix) |

## Setup

With Python 3.10 or newer:

```bash
git clone https://github.com/skai-research/Seeing-Through-Conflicts.git
cd Seeing-Through-Conflicts
python -m pip install -r requirements.txt
```

## Dataset generation

```bash
python build_dataset.py --output-dir data
```

Creates IH-text, IH-image, and IH-mixed from VerIH text examples in
`data/text`, `data/image`, and `data/mixed`.

## Training

In your GPU environment:

```bash
python -m pip install -e .
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

Edit [run.sh](run.sh) to choose your model, dataset, and GPU count, then run:

```bash
bash run.sh
```

Training hyperparameters are in [scripts/train_grpo.sh](scripts/train_grpo.sh).

## Turn text into an image

```bash
python render_text.py "Respond with three commas." --output prompt.png
```

Or use the function directly:

```python
from render_text import render_text

image = render_text("Respond with three commas.", "prompt.png")
```

## Acknowledgments

- Our training builds on [VerIH](https://github.com/skai-research/VerIH), [TinyZero](https://github.com/Jiayi-Pan/TinyZero), and [veRL](https://github.com/volcengine/verl).
- We use [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) Thinking base models.
- Our text examples and constraint verifiers build on VerIH, [RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval), and [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval).
