# Anomaly-Atoms (Prototype)

**In essence, we transform atoms from being regarded by VLMs as "relevant" to be with clear logical roles**

End-to-end **prototype** of the atom-based anomaly pipeline you described.

It implements:

1) **Frame captions (atoms)** via Qwen2.5-VL-3B (optional / pluggable).
2) **Presence scores** p_c(t) using CLIP image/text encoders.
3) **Inter-frame divergence** d_t and **boundary probability** b_t = σ(κ(d_t−δ)).
4) **Segment-weighted presence** \bar p_c(u) using π_{t→u} = Π_{s=t+1}^u (1−b_s).
5) **Atom role learning** with direct / indirect / **counter** roles:
   P_seg(e|u) = σ(γ_dir·logit D_e(u) + γ_ind·logit I_e(u) + γ_ctr·logit C_e(u) + b_e).

Training objective follows Eq.(8)-(9): video-level label y_{n,e} supervising union of frames U_n.

> ⚠️ This is a **research scaffold**: it runs with CPU/GPU and standard Python libs, but
> heavy models (Qwen/CLIP) require installation and GPU. You can also plug **mock captions**
> to debug the math without the models (see `--dry_caption`).

## Install

```bash
conda create -n atoms python=3.10 -y
conda activate atoms
pip install -r requirements.txt
```

If you want CUDA: install a matching PyTorch build first (see pytorch.org).

## Quick Start

1) **Extract frames** (1 fps by default):
```bash
python scripts/extract_frames.py --video /path/to/video.mp4 --out_dir data/frames --fps 1
```

2) **Run the pipeline** (computes CLIP presence, boundaries, segment features, then saves tensors):
```bash
python scripts/run_pipeline.py \  --frames_root data/frames \  --labels_csv data/frame_labels.csv \  --out_dir runs/demo \  --window 20 \  --delta 0.6 --kappa 8.0 \  --qwen_json /path/to/qwen_atoms.json --lambda_atoms 0.25 --lambda_resemble 0.1  
```
Note: the function of this part is to use video segments that each frame belongs to to determine the presence scores of atoms in each frame. 

Format of input --frames_root: a directory of subfolders, the name of each subfolder is the name of a video (except suffix). Within each subfolder are the frames. 

Format of input --labels_csv: a DataFrame, each row has two columns, the first column is the directory of a frame, the second column is 0/1 label determined by CGSGM. 

Format of input --qwen_json: a dict, each key is the directory of one frame, each value is a sub-dict with two keys-"atoms" and "raw", they value of "atoms" is the list of all atoms that VLM thinks relevant to the abnormal event in current frame, the value of "raw" is the concatenation of strings in the list in "atoms", with a start symbol "ATOMS: ".

In current moment, VLM only knows that these atoms are relevant to the event, and waits for train_role_learning.py to determine their roles. 

Format of output seg_presence.pt — segment-weighted atom presence (seg_presence.pt is a segment-aware, post-processed version of presence.pt) 

Format of output trainpack.pt — bundle for the trainer

{
  "pbar":  <Tensor (T, C)>,   # exactly the tensor saved in seg_presence.pt
  "labels": <Tensor (T,)>      # same as labels.pt
}

Format of output labels.pt — frame labels provided by input 

Format of output presence.pt → raw per-frame atom scores 

Format of output boundary.pt → per-frame boundary probabilities  

3) **Train role learning** (supervises video-level labels for 6 events):
```bash
python scripts/train_role_learning.py \  --manifest_csv runs/manifest.csv --epochs 60 --topk 10 --freeze_bias_epochs 5 
```

Artifacts:
- `runs/demo/presence.pt` – T×C frame presence.
- `runs/demo/boundary.pt` – length-T boundary probs b_t.
- `runs/demo/seg_presence.pt` – T×C segment-aggregated presence \bar p_c(u).
- `runs/demo/trainpack.pt` – tensors used by the trainer.

Configure atoms/prompts in `src/atoms/ontology.py` and CLIP backbone in `src/embeddings/clip_backend.py`.

## Notes
- Presence score p_c(t) is **cosine** between normalized image embedding E_V(v_t) and text embedding t_c, mapped to [0,1] by (cos+1)/2.
- Inter-frame divergence d_t = ||α_t − α_{t−1}||_2^2 with α_t=[p_c(t)]_c.
- Segment prob π_{t→u} uses accumulated (1-b_s).
- Indirect evidence I_e(u) implements pairwise gating over atom pairs (i<j).
- Counter evidence C_e(u) ≈ "no strong direct" AND "not two indirects":
  C_e(u) = Π_c (1 − ρ_dir_c,e·\bar p_c(u)) · Π_{i<j} (1 − ρ_ind_i,e·ρ_ind_j,e·\bar p_i·\bar p_j).

This repo aims to be **clear** and **editable** for your research, not final production code.
