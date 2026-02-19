#TODO import train_and_evaluate_iCARL
#TODO import train_and_evaluate_ER
#TODO import train_and_evaluate_DER
from src.utils import print_strategy

def general_training_loop(
    trainset,
    testset,
    feature_dim,
    device,
    memory_size,
    attack_patterns, # multiple patterns, array of array
    epochs, 
    strategy_name
    ):

    if strategy_name == "iCARL":
        train_and_evaluate = train_and_evaluate_iCARL
    elif strategy_name == "ER":
        train_and_evaluate = train_and_evaluate_ER
    elif strategy_name == "DER":
        train_and_evaluate = train_and_evaluate_DER
    else:
        print("Strategy not implemented")

    print_strategy(strategy_name)

    for scenario_id, attack_pattern in enumerate(attack_patterns):
        train_and_evaluate(
            scenario_id,
            trainset,
            testset,
            feature_dim,
            device,
            memory_size,
            attack_patterns, # single pattern
            epochs
        )