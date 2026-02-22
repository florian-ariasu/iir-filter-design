# Manual Design of Butterworth and Chebyshev-I Digital IIR Filters — with ECG Denoising

> [!IMPORTANT]
> This is a **project assignment** for the Signal Processing (PS — Procesarea Semnalelor) course (4th year, 1st semester).
- The implementation follows the classical **analog-prototype → bilinear-transform** workflow, coded manually in Python without relying on high-level design wrappers.
- The project was developed entirely **by hand**, validated on both a synthetic test signal and a real ECG recording.

---

## Project Overview

This project presents the complete manual design, implementation, and evaluation of **Butterworth** and **Chebyshev-I** low-pass digital IIR filters, starting from amplitude and frequency specifications. The designed filters are then applied to:

- A **synthetic test signal** composed of a low-frequency sinusoid, high-frequency sinusoidal interference, and additive white Gaussian noise.
- A **real ECG recording** (record 100, MLII lead) from the MIT-BIH Arrhythmia Database.

Magnitude, phase, and pole–zero responses are analyzed alongside a selectivity metric and region-of-convergence (ROC) indicator. Time- and frequency-domain results are used to compare the denoising performance of both filters.

---

## Repository Structure

```
src/
├── butterworth.py          # Manual Butterworth analog prototype + digital realization
├── chebyshev1.py           # Manual Chebyshev-I analog prototype + digital realization
├── main.py                 # Synthetic signal design, analysis, and comparison
├── test_data.py            # ECG signal loading and filter application
├── 100_ekg.csv             # ECG record 100 from MIT-BIH (MLII and V5 leads, 360 Hz)
├── sample_signal/          # Output figures — synthetic signal experiments
│   ├── filter_analysis_butterworth.png
│   ├── filter_analysis_chebyshev-i.png
│   ├── filter_comparison_transition_band.png
│   ├── signal_comparison_butterworth.png
│   └── signal_comparison_chebyshev1.png
└── ecg_signal/             # Output figures — ECG denoising results
    ├── ecg_butterworth.png
    └── ecg_chebyshev-i.png

docs/
└── butter_cheby_paper.pdf  # Accompanying project paper (design methodology & results)
```

---

## Design Methodology

### Filter Specifications

Each filter is specified by passband ripple δ_p, minimum stopband attenuation δ_s, passband edge f_p, stopband edge f_s, and sampling frequency F_s.

| Parameter | Synthetic Signal | ECG Signal |
|---|---|---|
| F_s | 50 kHz | 360 Hz |
| f_p | 10 kHz | 40 Hz |
| f_s | 14 kHz | 60 Hz |
| δ_p | 0.5 | 0.5 |
| δ_s | 34 | 34 |

### Butterworth Design

The filter order is derived from the classical Butterworth relation. Normalized poles are placed on a unit-radius circle in the left half s-plane, then scaled to the prewarped cutoff frequency. The result is a **maximally flat passband** with a monotonic roll-off.

### Chebyshev-I Design

The filter order is derived using the inverse hyperbolic cosine relation. Poles are distributed on an ellipse in the left half s-plane. This yields a **steeper transition band** than Butterworth at the cost of equiripple in the passband.

### Digital Realization

Both prototypes are converted to digital filters via the **bilinear transform** with frequency prewarping:

```
ω̃_p = (2 / T) · tan(Ω_p / 2),   T = 1 / F_s,   Ω_p = 2π · f_p / F_s
```

The analog-to-digital mapping is performed with `bilinear_zpk`, and coefficients are converted to transfer-function form H(z) = B(z) / A(z) via `zpk2tf`. Zero-phase filtering on the ECG signal is applied using `filtfilt`.

---

## Results

### Synthetic Signal

Both filters successfully attenuate the 20 kHz interference and broadband noise while preserving the 5 kHz useful sinusoid. The Chebyshev-I filter achieves a smaller selectivity factor, corresponding to a visibly sharper transition band.

| Filter | Order | Selectivity (F_ss ↓ is better) |
|---|---|---|
| Butterworth | 18 | higher |
| Chebyshev-I | 5 | lower ✓ |

### ECG Denoising

Applied to record 100 (MLII lead, 360 Hz). Both filters reduce high-frequency noise and preserve the P–QRS–T morphology without introducing noticeable distortion. Chebyshev-I produces slightly sharper high-frequency attenuation, consistent with the synthetic results.

---

## Running the Project

**Requirements:** Python 3.x with `numpy`, `scipy`, `matplotlib`, and `pandas`.

Install dependencies:
```bash
pip install numpy scipy matplotlib pandas
```

Run the synthetic signal experiment:
```bash
python src/main.py
```

Run the ECG denoising experiment:
```bash
python src/test_data.py
```

Output figures will be saved in `src/sample_signal/` and `src/ecg_signal/`.

---

## References

- S. K. Singh et al., "Computational Analysis of Butterworth and Chebyshev-I Filters Using Bilinear Transformation," *GRANTHAALAYAH*, vol. 10, 2022. [DOI: 10.29121/granthaalayah.v10.i6.2022.4571](https://doi.org/10.29121/granthaalayah.v10.i6.2022.4571)
- G. B. Moody and R. G. Mark, "The Impact of the MIT-BIH Arrhythmia Database," *IEEE Engineering in Medicine and Biology Magazine*, vol. 20, no. 3, pp. 45–50, May–Jun. 2001.

The full project paper is available in [`docs/butter_cheby_paper.pdf`](./docs/butter_cheby_paper.pdf).

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
