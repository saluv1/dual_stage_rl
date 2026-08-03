# Final package manifest

## Sequential commands

1. `commands/00_validate_phase1_and_ab.sh`
2. `commands/01_evaluate_lqr.sh`
3. `commands/02_train_td3.sh`
4. `commands/03_evaluate_td3_rk45.sh`
5. `commands/04_evaluate_td3_official.sh`
6. `commands/05_evaluate_ab_constraints.sh`

## Main implementation

- `env/dynamics.py`: official forward-Euler dynamics with optional RK45 check
- `bcbf/lqrgain.py`: official hover DLQR and Riccati matrix
- `backup_policy/replay_buffer.py`: real transitions and physical raw actions
- `backup_policy/td3.py`: exact terminal values and official-aligned TD3 update
- `backup_policy/train_modular.py`: indexed local training workflow
- `backup_policy/phase1/sampling.py`: official reset-library curriculum
- `official_phase1_evaluation/`: local NumPy/PyTorch port of Phase-I evaluation
- `bcbf/official_ab.py`: backup-flow sensitivity and `A(x), b(x)` matrices

## Outputs

- models: `Trained Models/001`, `002`, ...
- LQR: `evaluation/LQR Evaluation/`
- training plots: `evaluation/Training Progress/<index>/`
- RK45: `evaluation/TD3 Evaluation RK45/`
- official protocol: `evaluation/TD3 Evaluation Official/`
- constraints: `evaluation/A-B Constraint Evaluation/`

The official reset-library asset is bundled at
`official_phase1_evaluation/assets/reset_library.pkl`.
