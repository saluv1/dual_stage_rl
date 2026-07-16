# dual_stage_rl
Undergraduate Independence Research in SNU

Repository maintainers 
[Jaeyoung Lee](https://saluv1.github.io/), Jungwon Kim<br>
Seoul National University

This branch provides a moderately refactored version of ps2_rl to facilitate easier on hardware-environment

## High-Level Structure

## Environment Setup

## Phase I
The main purpose of Phase I of PS2-RL is to apply the concept of Backup Control Barrier Functions and the Safe Arrival Policy to train the backup policy, which will then be used to derive the implicit control-invariant set.

Progress
- Dynamics (Completed)
- LQR Base Controller (Completed)
- Base set (On-Going)
- Safe Arrival Policy (RL Algorithm) (To-do)

## Phase II
it should solve optimization problem via qpax (qp solver) in jax
  <br/>so, I bring three main codes that will be combined into one architecture. 
  <br/> [SAC in jax](https://github.com/aliciafmachado/sac) 
  <br/>[hardnet in pytorch](https://github.com/azizanlab/hardnet) 
  <br/>[making trajectory via BCBF idea](https://github.com/davidvwijk/DR-bCBF) 
