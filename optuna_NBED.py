import optuna
import NBED
from typing import Dict, Any, Optional

class OptunaOptimizer:
    def __init__(self, study_name: str = "nbp_optimization", storage: Optional[str] = None):
        self.study_name = study_name
        self.storage = storage
        
    def suggest_params(self, trial: optuna.Trial, code_type: str) -> Dict[str, Any]:
        """Suggest hyperparameters"""
        if code_type == 'toric':
            return {
                "batch_size": trial.suggest_categorical("batch_size", [60, 80, 100, 120, 140, 160]),
                "num_batch": trial.suggest_int("num_batch", 50, 200, step=5),
                "learning_rate": trial.suggest_float("learning_rate", 0.1, 1.0, step=0.1),
                "num_points": trial.suggest_int("num_points", 4, 10),
                "epsilon0": trial.suggest_float("epsilon0", 0.3, 0.6, step=0.025),
                "epsilon1": trial.suggest_float("epsilon1", 0.04, 0.06, step=0.01),
            }
        else:  # GB
            return {
                'batch_size': trial.suggest_categorical('batch_size', [60, 80, 100, 120, 140, 160, 200, 240]),
                'num_batch': trial.suggest_int('num_batch', 750, 3000),
                'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
            }
    
    def objective(self, trial: optuna.Trial, n: int, k: int, m: int, code_type: str, 
                  n_iterations: int, error_weights: tuple) -> float:
        """Objective function for Optuna optimization"""
        params = self.suggest_params(trial, code_type)
        
        # Create decoder with suggested parameters
        decoder = NBED.init_and_train(n, k, m, n_iterations, error_weights, 
                                           code_type, params, name = "optuna")
        
        final_loss = decoder.final_loss
        return final_loss
    
    #It would also be possible to write a wrapper for the C++ evaluation code, which might be a better metric
    def optimize(self, n_trials: int, **kwargs) -> optuna.Study:
        """Run optimization study"""
        study = optuna.create_study(
            direction="minimize",
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True
        )
        
        study.optimize(
            lambda trial: self.objective(trial, **kwargs),
            n_trials=n_trials
        )
        
        return study

# Usage example:
def run_optimization():
    optimizer = OptunaOptimizer("toric_optimization")
    
    study = optimizer.optimize(
        n_trials=50,
        n=128, k=2, m=384, code_type='toric',
        n_iterations=18, error_weights=(4,7)
    )
    
    print("Best parameters:", study.best_params)
    print("Best loss:", study.best_value)
