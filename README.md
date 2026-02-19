# cil-intrusion-detection
Cybersecurity project for year 2025/2026 at UniBo

# TASKS for 18/02/2026
* [ ] pack also UNSW_NB15 in a similar way to CIC_IDS_2017 so that it's usable for training too --> exporting as .csv
  
* [ ] finish the three strategies: icarl, dar and er
  * [x] check that you're using the correct loss function
  * [ ] up-to-normalisation at the beginning of each task @Jeanritah
  * [ ] correct scenario and task definition -- first has one extra class the benign one, and benign class must always be class 0 @Jeanritah
  * [ ] check that the buffer contains just the indices to the original dataset and not a copy of the element, or modified element
  * [ ] after each task compute accuracy and f1-score, plot the confusion matrix and export it to png in the `results/confusion_matrices/ folder`@Jeanritah, @m-mejiaj
* [x] After the training loop compute the average accuracy and the forgetting measure 
  
* [ ] use common method to save results to json

* [ ] transform the training loop in a specific training method that respects the structure of the general training method

* [ ] common feature dimension 128

* [ ] add weighted samples to prevent bad learning with few samples

* [ ] Common output
```txt
Strategy <strategy name> ========

    === Scenario [1,1,1,1,1] ===

    --- Task 1 ---
    
    New attacks: [benign, dos]
    Seen so far: [portscan]

    accuracy: 0.00
    macro-f1: 0.00

    --- Task 2 ---

    --- Task n ---
    
    average accuracy:
    forgetting
```

* [ ] 2017 - 7 attacks, 2015 - 8 attacks, strategies to try [1,1,1,...,1]; 
    2017: [3,4] 2015: [4,4];  2017: [2, 2, 3] 2015: [2, 2, 4]

* [ ] methods to make comparison graphs for the report

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
