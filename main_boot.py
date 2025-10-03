import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from run_boot_traj import run_bootstrap_for_all_trajectories # импорт старой функции

if __name__ == "__main__":
    # Параметры
    npz_path = "sgd_1024_trajs_2.npz"
    base_seed = 12345
    N_boot = 256
    beta_a = 0.5
    beta_b = 2.0
    num_workers = 8
    batch_size = 32
    out_path = "theta_bar_boot_1024_2.npz"

    # Загружаем данные из файла
    data = np.load(npz_path)
    X_all = data['X_all']
    y_all = data['y_all']
    c0 = float(data['c0'])
    k0 = float(data['k0'])
    step = float(data['gamma'])  # gamma в старом файле = step

    # Запуск bootstrap
    OUT = run_bootstrap_for_all_trajectories(
        base_seed=base_seed,
        X_all=X_all,
        y_all=y_all,
        c0=c0,
        k0=k0,
        step=step,
        N_boot=N_boot,
        beta_a=beta_a,
        beta_b=beta_b,
        num_workers=num_workers,
        batch_size=batch_size,
        out_path=out_path
    )
    print("Saved to", OUT)