from traj_funcs import run_all_trajectories

if __name__ == "__main__":
    OUT = run_all_trajectories(
        n_trajectories=1024,
        N=50000,
        p=5,
        sigma=0.02,
        uniform_low=-1.0,
        uniform_high=3.0,
        base_seed=12345,
        c0=200.0,
        k0=20000.0,
        step=0.85,
        num_workers=8,
        batch_size=32,
        out_path="sgd_1024_trajs_linreg.npz"
    )
    print("Saved to", OUT)
