# Data splitting and reported accuracy in image-based ECG arrhythmia classification

A replication study. A small convolutional network is trained on beat images from the
MIT-BIH Arrhythmia Database under two data-splitting protocols, with the split as the
only variable:

- **Protocol A** — beat-level random split (beats from the same patient may appear in
  both training and test sets)
- **Protocol B** — inter-patient split (DS1 → DS2, de Chazal et al., 2004)

The model and hyperparameters are taken unchanged from an undergraduate honours project
that reported over 80 per cent accuracy under Protocol A.

This work **replicates a known result**. The inflation caused by beat-level splitting was
established by de Chazal et al. (2004) and has been documented repeatedly since. The
contribution here is a controlled, reproducible demonstration on an image-based pipeline,
with full code and per-class metrics.

## Requirements

```
numpy
wfdb
pillow
torch          # for torch_train.py
jax, optax     # for train.py (alternative implementation)
```

## Reproducing

1. Download the MIT-BIH Arrhythmia Database (v1.0.0) from PhysioNet:

   ```
   wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/
   ```

2. Build the beat-image dataset (writes `X.npy`, `y.npy`, `rid.npy`). Set `SRC` in
   `build_dataset.py` to the downloaded record directory:

   ```
   python build_dataset.py
   ```

   This produces 100,694 beats from the 44 non-paced records (102, 104, 107 and 217 are
   excluded per AAMI practice), lead MLII, rendered as 120×120 grayscale images from a
   0.7 s window centred on each annotated R peak. Every beat keeps its record identifier.

3. Train and evaluate under both protocols:

   ```
   for s in 0 1 2 3 4; do
     python torch_train.py --protocol A --seed $s --data .
     python torch_train.py --protocol B --seed $s --data .
   done
   ```

Results are written as JSON, one file per protocol and seed, containing overall accuracy,
per-class sensitivity, positive predictive value and F1, macro-F1 over N/S/V/F, and the
confusion matrix.

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Builds the beat-image dataset from raw MIT-BIH signals |
| `torch_train.py` | PyTorch implementation (canonical) |
| `train.py` | JAX implementation used for the initial runs |

## Notes on class Q

Class Q (unknown/paced) has only 15 beats across the 44 records once paced records are
excluded. It is reported for completeness but excluded from macro-F1, following common
practice.

## Citation of the original data

Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and
Biol 20(3):45-50 (2001). Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet.
Circulation 101(23):e215-e220 (2000).
