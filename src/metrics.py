import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, accuracy_score

def accuracy(y_true, y_pred):

  return accuracy_score(y_true, y_pred)

def macro_f1(y_true, y_pred):

  return f1_score(y_true, y_pred, average='macro')

def average_accuracy(task_accuracies):

  """
  task_accuracies: list of accuracies up to current task
  """
  return np.mean(task_accuracies)

def compute_forgetting(results):
    """
    results: a matrix of shape (num_tasks, num_tasks) where results[i][j] is the accuracy on task j after training on task i.
    """
    # Ensure results is a numpy array for easy slicing
    results = np.array(results)
    num_tasks = len(results)
    
    # List to store the forgetting for each task
    forgetting_per_task = []

    # Iterate over all tasks (starting from task 1)
    for k in range(1, num_tasks):
        f_k = 0  # Initialize forgetting for task k
        for j in range(k):  # Compare task k with all previous tasks (0 to k-1)
            best_past = np.max(results[:k])  # Maximum accuracy from all previous tasks
            current = results[k]  # Current performance on task k
            f_k += (best_past - current)  # Forgetting is the difference

        # Compute the average forgetting for task k
        f_k /= k
        forgetting_per_task.append(f_k)
    
    # Return the average forgetting across all tasks
    return np.mean(forgetting_per_task)

def save_confusion_matrix(cm, class_names, out_path, title=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title or "Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close() 

def compute_cm(y_true, y_pred, seen_classes, show_plot=True):
  cm = confusion_matrix(y_true, y_pred)
  if show_plot:
    disp = ConfusionMatrixDisplay(cm, display_labels=seen_classes)
    disp.plot()

    # Rotate x-axis labels
    plt.xticks(rotation=45)  # change 45 to 90 if needed
    plt.show()

  return cm

def average_accuracy(f1_history):

  """
  f1_history: list of macro_f1 up to current task
  """
  return np.mean(f1_history)