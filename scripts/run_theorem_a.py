import torch
from configs.train_configs import DecoupledTrainModelConfig
from dynamics.pretrain_icl_powerlaw import make_normalized_powerlaw_problem, train_model_with_checkpoints

cfg = DecoupledTrainModelConfig(
    d=32,
    P_tr=32,
    P_test=32,
    B=512,
    N=32,
    L=4,
    beta_model=1.0,
    gamma=1.0,
    T=6000,
    lr=0.125,
    lamb=1.0e-14,
    alpha=0.0,   # flat normalized spectrum => isotropic code regime
    beta=1.75,
    sigma=0.4,
    unrestricted=False,
    online=True,
    sample_mode="spec",
)

device = "cuda"
dtype = torch.float64  # use float64 for theorem-A auditing

spec, w_star = make_normalized_powerlaw_problem(
    cfg.d, cfg.alpha, cfg.beta, device=device, dtype=dtype
)

bundle = train_model_with_checkpoints(
    cfg,
    spec=spec,
    w_star=w_star,
    checkpoint_steps=(0, cfg.T // 2, cfg.T),
    debug_seed=1234,
    debug_batch_size=64,
    device=device,
    dtype=dtype,
)

for step in sorted(bundle["checkpoints"]):
    print(f"\nstep={step}")
    for k, v in bundle["checkpoints"][step]["summary"].items():
        print(f"{k}: {v:.6e}")