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
    results: 2D list or numpy array
             shape (num_tasks, num_tasks)
             results[t][i] = metric on task i after training task t
    """

    results = np.array(results)
    num_tasks = results.shape[0]

    forgetting_per_task = []

    for t in range(1, num_tasks):
        f_t = 0
        for i in range(t):
            best_past = np.max(results[:t, i])
            current = results[t, i]
            f_t += (best_past - current)

        f_t /= t
        forgetting_per_task.append(f_t)

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