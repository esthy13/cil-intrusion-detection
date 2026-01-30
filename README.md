# cil-intrusion-detection
Cybersecurity project for year 2025/2026 at UniBo

## Submission
Project submission on the 30th of january 
- form: https://docs.google.com/forms/d/e/1FAIpQLSeVGfQmJF3rCgumOtC069-qDpiXhNg_7yOc4-E_IieJE1jrMw/viewform
- Code (it must be well documented with a README)
- Report
- Presentation

## Questions:
- Just one dataset or both datasets?

## Tutor for the project: 
isabella.marasco4@gmail.com

# WORKFLOW
1. Write everything in a personal jupiter notebook
2. Once tested and after group review put the code in one of the following modules:
    - model.py
    - strategies.py
    - training.py
    - metrics.py - evaluation
    - preprocessing.py
    - scenarios.py

## Steps:
1. Realize an IDS CIL scenario based using PyTorch.
    - dataset exploration -- Jean
    - reading resources and checking how to implement the system with a code example -- Esther & Margarita
2. Integrate these three   strategies for mitigating catastrophic forgetting: iCarl, DarkExperience, and ER.- Evaluate how the size of the incremental task (i.e., the number of new attack classes introduced in each phase) affects performance. Different **scenarios** will be compared (e.g., 10 total attacks introduced as 1+1+1... vs. 5+5 vs. 2+3+5) to  understand the system's sensitivity to the frequency and size of updates, the benign traffic is always present.
3. [Few-Shot Scenario Evaluation (Optional): extend the analysis to investigate how the system performs when new attack classes are introduced with a very limited number of samples (few-shot learning), a realistic scenario for emerging threats.]

## Evaluation Metrics:
- Accuracy and average accuracy- Forgetting: measures the average performance degradation on classes learned in previous tasks after training on new tasks

## Datasets:
- https://www.unb.ca/cic/datasets/ids-2017.html
- https://research.unsw.edu.au/projects/unsw-nb15-dataset

## Reources:
- https://ieeexplore.ieee.org/abstract/document/10444954?casa_token=Dk7EHtaXWGEAAAAA:C8wE_rD51BckgXNqxkXW-H5eFG2vy8OxCILQiUvD2BANfJ6mHYZ36_IsUXIuZs3eMADSNA

- https://ieeexplore.ieee.org/document/10599804/

##