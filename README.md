# cil-intrusion-detection
Cybersecurity project for year 2025/2026 at UniBo

Cyber

# Structure of the repository
```txt
cil-intrusion-detection/
├─ README.md
├─ data/
│  ├─ raw/            # original datasets (not tracked in git)
│  ├─ processed/      # preprocessed datasets (optional)
│    ├─2017.zip
│    └─2015.zip --> when unzipping you get the following structure:
│                       ├─ 2017/
│                           ├─ train/ # one file for each class 
│                           └─ set/
├─ notebooks/
│  ├─ cleaning_2015.ipynb # and dataset creation 
│  ├─ cleaning_2017.ipynb
│  └─ comparison.ipynb
├─ examples/
│  ├─ cleaning_2015.ipynb # and dataset creation 
│  ├─ cleaning_2017.ipynb
│  └─ comparison.ipynb
├─ src/
│  ├─ dataset.py         # class dataset
│  ├─ model.py        # class model + (build task?)
│  ├─ train.py        # general training loop
│  ├─ metrics.py  
│  ├─ iCarl.py        
│  ├─ ER.py             
│  └─ DER.py
├─ weights/
│  ├─ icarl.weights.h5
│  ├─ ER.weights.h5
│  └─ DER.weights.h5
├─ results/
│  ├─ training/
│  ├─ confusion_matrices/
│  └─ comparisons/
├─ artifacts/ 
│  ├─ report.pdf
│  └─ presentation.pdf
└─ .gitignore
```

## Submission
Project submission on the 30th of january 
- form: https://docs.google.com/forms/d/e/1FAIpQLSeVGfQmJF3rCgumOtC069-qDpiXhNg_7yOc4-E_IieJE1jrMw/viewform
- Code (it must be well documented with a README)
- Report
- Presentation

## Tutor for the project: 
isabella.marasco4@gmail.com

## Datasets:
- https://www.unb.ca/cic/datasets/ids-2017.html
    - https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset
- https://research.unsw.edu.au/projects/unsw-nb15-dataset
    - https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fz5025758%5Fad%5Funsw%5Fedu%5Fau%2FDocuments%2FUNSW%2DNB15%20dataset%2FCSV%20Files&viewid=f8d1dec5%2Dcd5f%2D42ae%2D8b06%2D2fece580c74a

## Reources:
- https://ieeexplore.ieee.org/abstract/document/10444954?casa_token=Dk7EHtaXWGEAAAAA:C8wE_rD51BckgXNqxkXW-H5eFG2vy8OxCILQiUvD2BANfJ6mHYZ36_IsUXIuZs3eMADSNA

- https://ieeexplore.ieee.org/document/10599804/
