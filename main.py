#Import packages
from pathlib import Path

from src.icarl import train_icarl
from src.utils import set_seed, build_parser

import os
print(os.getcwd())


# directory
Root = Path(__file__).resolve().parent
folder_dataset = Root /  "data" / "processed"
folder_json = Root / "results" / "training"
folder_json.mkdir(parents=True, exist_ok=True)
folder_cm = Root / "results" / "confusion_matrices"
folder_cm.mkdir(parents=True, exist_ok=True)

#TODO: you can just call the general training loop
def get_strategy(strategy_name, model, criterion, **strategy_kwargs):
    if strategy_name == 'er':
        return ER(model=model, criterion=criterion, **strategy_kwargs)
    elif strategy_name == 'icarl':
        return iCaRL(model=model, criterion=criterion, **strategy_kwargs)
    elif strategy_name == 'der':
        return DER(model=model, criterion=criterion, **strategy_kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

def main():

    set_seed()
    args, strategy_kwargs = build_parser()
    strategy_name = args.strategy
    scenarios = list(args.scenarios)
    
    if strategy_name == "icarl":
        for i in scenarios:
            train_icarl(root_dir_dataset = folder_dataset,
                        dataset_name = args.dataset,
                        out_path = folder_cm,
                        json_path = folder_json,
                        label_col = "Label" if args.dataset == '2017' else 'attack_cat', # review
                        feature_dim = args.feature_dim,
                        memory_size = args.memory_size,
                        epochs = args.epochs,
                        batch_size = args.batch_size,
                        lr = args.lr,
                        attack_pattern = scenarios[i],
                        **strategy_kwargs)
            
    elif strategy_name == "er":
        for i in scenarios:
            train_er(

                **strategy_kwargs)
    
    elif strategy_name == 'der':
        for i in scenarios:
            train_der(

                **strategy_kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

#---------------------------------------------------------

if __name__ == "__main__":

    main()
    #in order to run the program also from the command line and not just from colab