#Import packages
from pathlib import Path

from src.icarl.icarl import train_icarl
from src.icarl.utils import set_seed, build_parser
from src.train_DER import train_and_evaluate_DER

import os
print(os.getcwd())

# directory
Root = Path(__file__).resolve().parent
folder_dataset = Root /  "data" / "processed"
folder_results = Root / "results" / "training"
folder_results.mkdir(parents=True, exist_ok=True)

def main():

    set_seed()
    args, strategy_kwargs = build_parser()
    strategy_name = args.strategy
    scenarios = list(args.scenarios)
    
    if strategy_name == "icarl":
        for i in scenarios:
            train_icarl(strategy_name=strategy_name,
                        dataset_path=folder_dataset,
                        dataset_name=args.dataset,
                        ouput_path=folder_results,
                        feature_dim=args.feature_dim,
                        memory_size=args.memory_size,
                        epochs = args.epochs,
                        batch_size = args.batch_size,
                        lr = args.lr,
                        attack_pattern = scenarios[i],
                        **strategy_kwargs)
                       
    elif strategy_name == "er":
        print('er')
    elif strategy_name == "der":
        print('der')
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

#---------------------------------------------------------

if __name__ == "__main__":

    main()
    #in order to run the program also from the command line and not just from colab