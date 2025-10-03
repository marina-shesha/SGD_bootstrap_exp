from traj_funcs_log  import run_all_trajectories

if __name__ == "__main__":
    OUT = run_all_trajectories(
        n_trajectories=1024,
        N=50000,
        N_val=10000,
        p=5,
        uniform_low=-1.0,
        uniform_high=3.0,
        base_seed=12345,
        c0=200.0,
        k0=20000.0,
        step=0.85,
        reg=0.0001,
        num_workers=8,
        batch_size=32,
        out_path="sgd_1024_trajs_logreg.npz"
    )
    print("Saved to", OUT)
