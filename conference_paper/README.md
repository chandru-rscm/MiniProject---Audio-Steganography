# StegoWave: International Conference Submission & Academic Assets

This directory contains the finalized academic publication artifacts for **StegoWave: A Capacity-First Multimodal Audio Steganography Framework Using Adaptive Compression, Authenticated Cryptography, and Two-Bit LSB Embedding**.

---

## 📁 Directory Structure

```text
conference_paper/
├── README.md                              # Submission guide & overview (this file)
├── StegoWave_Conference_Paper_Final.docx  # Final 8-page master Word document (IOP/IEEE compliant)
├── StegoWave_Final_Paper.tex              # Complete compilation-ready IEEEtran LaTeX source code
├── fig01_dataflow.png                     # System data-flow diagram (Layer 1, 2, 3)
├── fig02_acac_pipeline.png                # ACAC binary-search Lanczos compression pipeline
├── fig03_bundle_assembly.png              # Pure-Python AES-256-GCM authenticated bundle assembly
├── fig04_radar_chart.png                  # Qualitative radar chart resolving the steganographic trilemma
└── figures/                               # Backup directory containing high-resolution diagram figures
```

---

## 🚀 How to Build & Submit the Paper

### Option 1: Overleaf (Recommended for LaTeX)
1. Go to [Overleaf.com](https://www.overleaf.com) and click **New Project** $ightarrow$ **Blank Project**.
2. Upload `StegoWave_Final_Paper.tex`, `fig01_dataflow.png`, `fig02_acac_pipeline.png`, `fig03_bundle_assembly.png`, and `fig04_radar_chart.png` directly to your Overleaf workspace.
3. Click **Recompile** — the project will instantly build a stunning, publication-ready PDF conforming strictly to the IEEEtran 10pt conference standard.

### Option 2: Local LaTeX Compilation (`pdflatex` / `TeXstudio`)
If you have a local LaTeX distribution (such as **MiKTeX** or **TeX Live**) installed:
```bash
cd conference_paper
pdflatex StegoWave_Final_Paper.tex
pdflatex StegoWave_Final_Paper.tex  # Run twice for cross-reference & citation resolution
```

### Option 3: Microsoft Word Submission
If submitting directly via `.docx` to conferences requiring Word templates (such as *IOP Journal of Physics: Conference Series*):
* Open `StegoWave_Conference_Paper_Final.docx` directly in Microsoft Word.
* The document strictly adheres to an **8-page budget**, contains **consecutively numbered Cambria Math equation tables (1–4)**, **professional dark teal `#002B36` benchmarking tables**, and **complete IEEE citations [1]–[22]**.

---

## 📊 Summary of Technical & Empirical Highlights

* **Payload Capacity:** Up to **$20\text{ MB}$** visual payloads embedded within $1\text{--}10\text{ MB}$ audio carriers via ACAC ($141.6\times$ capacity multiplier over transform baselines).
* **Cryptographic Security:** Pure-Python **AES-256-GCM** authenticated cryptography ($100,000$ PBKDF2 rounds) with a zero-leakage **authenticate-then-decrypt hard gate**.
* **Lossless Reversibility:** **$\text{BER} = 0.00000$** across all 60 stratified benchmark evaluations (`DIV2K`, `TIMIT`, `GTZAN`, `EBU-SQAM`).
* **Acoustic Transparency:** Audio carrier fidelity remains strictly $\ge 61.7\text{ dB}$ PSNR across all test cases.
* **Edge Feasibility:** Sub-millisecond desktop execution ($71.75\text{ ms}$) and verified embedded hardware viability on $100\text{ MHz}$ ARM Cortex-M4 microcontrollers ($<244\text{ ms}$).
