# DDSP-Violin: Unsupervised Recovery of Bowing Controls

Code for **"DDSP-Violin: Unsupervised Recovery of Bowing Controls Through Interpretable Source Parameters"** (AES Convention 2026). DDSP-Violin is a differentiable bowed-string synthesiser whose harmonic-amplitude encoder drives a brightness exponent α and a bow position over node orders k ∈ {0, 2, 3, 4, 5}. Trained to reconstruct audio, α follows bow force and speed and the node order is recovered. The model extends [DDSP-Violin](https://github.com/JoaoerDuarte/DDSP-Violin-EUSIPCO2026) (EUSIPCO 2026).

## Setup

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

## Data

The synthetic dataset is not included. The code reads 4 s, 16 kHz mono wav files from `dataset/audio/{train,test}/` (or the folder set by `data: {audio_dir: ...}` in a config), named like `G_m000.00p_F0.5953N_vb0.6231v_z0.2505x_k4_r003945.wav` (string, semitones above the open string, node order after `_k`). Evaluation also reads `dataset/manifests/test_k{0,2,3,4,5}.csv` with the columns `filename, string, semitone, F, vb, z0, k_class`.

## Training

```bash
python train.py --config ddsp_violin
```

`configs/` holds one config per row of the paper's tables. For another seed, copy a config to `<name>_v2.yaml` and train it. `configs/baseline.yaml` is the DDSP baseline of the EUSIPCO paper, with a free harmonic distribution. `resonance_type: ar` or `arma` under `model:` replaces the FIR body filter by an IIR one, with `resonance_ar_order` and `resonance_ma_order`.

## Evaluation

```bash
python preprocess.py --config ddsp_violin --split test
python evaluate.py
```

Writes both result tables, mean ± std over seeds, to `output/evaluate/results.txt`. `python evaluate_filter.py --responses <folder> runs/<config> ...` prints the body-filter errors of the EUSIPCO paper (MC-LSD, tilt and resonance error) against measured body responses, which are not included here.

## Citation

```bibtex
@inproceedings{duarte2026recovery,
  title={{DDSP-Violin}: Unsupervised Recovery of Bowing Controls Through Interpretable Source Parameters},
  author={Duarte, Jo{\~a}o and Mignot, R{\'e}mi and McDermott, James and O'Leary, Se{\'a}n},
  booktitle={Audio Engineering Society Convention},
  year={2026}
}
```

## Acknowledgments

Built upon [acids-ircam/ddsp_pytorch](https://github.com/acids-ircam/ddsp_pytorch). Original DDSP framework: Engel et al. (ICLR 2020). Bowed-string physical model: Demoucron (2008).

This work was conducted with the financial support of the Research Ireland Centre for Research Training in Digitally-Enhanced Reality (d-real) under Grant No. 18/CRT/6224. Additional support for co-supervision was provided by the project PostGenAI@Paris (ANR-23-IACL-0007, CAP AI-MADE).

## License

MIT
