# Automatic Detection of PQRST Fiducial Points in ECG Signals

> Bachelor's Thesis Project

## Project Description

This project focuses on the automatic detection of **PQRST fiducial points** in electrocardiogram (ECG) signals. ECG is a widely used non-invasive diagnostic technique that records the electrical activity of the heart and provides valuable information about cardiac function.

Accurate localization of the P wave, QRS complex, and T wave is essential for cardiac analysis. However, ECG signals are often affected by noise, baseline wander, measurement artifacts, and physiological variations. This repository investigates several approaches for robust PQRST detection, combining classical signal processing techniques with machine learning methods.

## Objective

The primary objective of this project is to develop and evaluate methods for the automatic detection of PQRST fiducial points in ECG recordings. Three different approaches will be investigated and compared:

- Pan-Tompkins algorithm
- Wavelet transform-based detection
- Machine learning-based detection

The methods will be evaluated on publicly available ECG databases using common performance metrics and compared with existing approaches reported in the literature.

## Status / Roadmap
 
This project is under active development as part of a BSc thesis. Current progress:
 
- [x] Literature review
- [x] ECG preprocessing pipeline
- [x] R-peak detection — Pan-Tompkins algorithm
- [x] Full PQRST delineation — heuristic rules around R-peak
- [ ] Wavelet transform-based PQRST detection
- [ ] Machine learning-based PQRST detection
- [ ] Evaluation on MIT-BIH Arrhythmia Database
- [ ] Evaluation on LUDB
- [ ] Comparison with literature results
- [ ] Performance benchmarking
- [ ] Final thesis documentation

*(This checklist is updated as the project progresses.)*

## Features

- ECG signal preprocessing and denoising
- R-peak detection using the Pan-Tompkins algorithm + PQRST delineation using heuristic rules
- Wavelet transform-based PQRST detection
- Machine learning-based PQRST detection
- Quantitative evaluation on benchmark datasets
- Performance comparison between the implemented methods
- Visualization of detected fiducial points

## Project structure

```
ecg-waveform-fiducials/
├── configs/            # Configuration files
├── data/               # ECG databases (gitignored)
├── scripts/            # Utility scripts
├── src/                # Main Python package
├── notebooks/          # Exploratory work
├── results/            # Generated tables, figures
└── thesis/             # Thesis source files
```

## Installation

### Clone the GITHUB repository
```bash
git clone https://github.com/gorozdi-attila/ecg-waveform-fiducials
cd ecg-waveform-fiducials
```

### Setup virtual environment on Linux / macOS
```bash
python -m venv venv
source venv/bin/activate

pip install -e .
```

### Setup virtual environment on Windows
```bash
python -m venv venv
venv\Scripts\activate

pip install -e .
```

### Dependencies

See `pyproject.toml` for the complete dependency list.

- Python 3.10+
- `PyYAML`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `PyWavelets`
- `torch`
- `scikit-learn`
- `wfdb`

## Datasets

The methods will be evaluated using two publicly available databases:
- **[MIT-BIH Arrhythmia Database](https://www.physionet.org/content/mitdb/1.0.0/ 'MIT-BIH Arrhythmia Database')** - provides expert-annotated R-peak locations and serves as a standard benchmark for evaluating R-peak detection algorithms.
- **[Lobachevsky University Electrocardiography Database](https://physionet.org/content/ludb/1.0.1/ 'Lobachevsky University Electrocardiography Database')** - provides manually annotated P, QRS, and T wave delineations, enabling full PQRST detection evaluation.

### Downloading the datasets

The databases are not included in this repo (due to its size). To download them:

```bash
python scripts/download_database.py mitdb
python scripts/download_database.py ludb
```

This downloads the records into the `data/mitbih/` and `data/ludb/` folders using the `wfdb` package.

## Planned Evaluation
 
The implemented methods will be evaluated using widely adopted performance metrics for ECG detection and delineation.
 
### Detection Performance
 
The following metrics will be used to assess the accuracy of heartbeat and waveform detection:
 
- **Sensitivity (Se)** / **Recall**
- **Positive Predictive Value (PPV)** / **Precision**
- **F1 Score**
- **Detection Error Rate (DER)**
### Localization Accuracy
 
To evaluate the temporal precision of detected fiducial points, the following metric will be used:
 
- **Mean Absolute Timing Error (MATE)**

The evaluation will be conducted separately for **R-peak detection** using the **MIT-BIH Arrhythmia Database** and for **complete PQRST delineation** using the **Lobachevsky University Electrocardiography Database (LUDB)**. This separation enables a comprehensive comparison of the three implemented approaches with respect to both detection performance and temporal localization accuracy.
 
Detailed quantitative results, summary tables, and visualizations will be generated and stored in the `results/` directory.
 
## Evaluation Protocol
 
All methods are evaluated under identical preprocessing and sampling conditions. Train/test splits (for ML methods) follow record-wise separation to avoid patient leakage.

## References

1. Pan, J., & Tompkins, W. J. (1985). *A Real-Time QRS Detection Algorithm*. **IEEE Transactions on Biomedical Engineering**, 32(3), 230–236.

2. Martínez, J. P., Almeida, R., Olmos, S., Rocha, A. P., & Laguna, P. (2004). *A Wavelet-Based ECG Delineator: Evaluation on Standard Databases*. **IEEE Transactions on Biomedical Engineering**, 51(4), 570–581.

3. Kiranyaz, S., Ince, T., & Gabbouj, M. (2016). *Real-Time Patient-Specific ECG Classification by 1-D Convolutional Neural Networks*. **IEEE Transactions on Biomedical Engineering**, 63(3), 664–675.

4. Moody, G. B., & Mark, R. G. (2001). *The MIT-BIH Arrhythmia Database*. **IEEE Engineering in Medicine and Biology Magazine**, 20(3), 45–50.

5. Kalyakulina, A., et al. (2020). *LUDB: A New Open-Access Validation Tool for Electrocardiogram Delineation Algorithms*. **IEEE Access**, 8, 186181–186190.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

Görözdi Attila — BSc in Computer Science 

Széchenyi István University, Győr

2026