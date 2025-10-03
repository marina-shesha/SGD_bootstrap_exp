import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def run_bootstrap_trajectory(base_seed, traj_id, boot_id, X, y, alphas, beta_a=2.0, beta_b=2.0):
    N, p = X.shape
    seed = int(base_seed) + int(traj_id * 1000000) + int(boot_id)
    rng = np.random.default_rng(seed)

    # Генерируем шум для alpha
    alpha_noise = rng.beta(beta_a, beta_b, size=N)
    E_beta = beta_a / (beta_a + beta_b)
    D_beta = (beta_a * beta_b) / ((beta_a + beta_b)**2 * (beta_a + beta_b + 1.0))
    alpha_noise = alpha_noise / np.sqrt(D_beta) + 1 - E_beta / np.sqrt(D_beta)
    alphas_noisy = alphas * alpha_noise

    theta_hat = np.zeros(p)
    theta_bar = np.zeros(p)
    theta_bar_history = np.zeros((N + 1, p))
    theta_bar_history[0] = theta_bar.copy()

    for n in range(1, N + 1):
        xn = X[n - 1]
        yn = y[n - 1]
        alpha_n = alphas_noisy[n - 1]

        residual = yn - xn.dot(theta_hat)
        grad = -2.0 * residual * xn

        theta_hat -= alpha_n * grad
        theta_bar = ((n - 1) * theta_bar + theta_hat) / n
        theta_bar_history[n] = theta_bar.copy()

    return traj_id, boot_id, theta_bar_history[10000::10000]

def run_batch_trajectories(base_seed, batch_ids, traj_id, X, y, alphas, beta_a, beta_b):
    """
    Запуск пакета bootstrap-траекторий, чтобы снизить накладные расходы на pickle.
    """
    results = []
    for boot_id in batch_ids:
        results.append(
            run_bootstrap_trajectory(base_seed, traj_id, boot_id, X, y, alphas, beta_a, beta_b)
        )
    return results

def run_bootstrap_for_all_trajectories(base_seed,
                                       X_all,
                                       y_all,
                                       c0,
                                       k0,
                                       step,
                                       N_boot=100,
                                       beta_a=2.0,
                                       beta_b=2.0,
                                       num_workers=None,
                                       batch_size=16,
                                       out_path="sgd_trajectories.npz"):
    """
    Для каждой исходной траектории запускает N_boot траекторий с шумными шагами
    и возвращает массив theta_bar_boot_mean[n_trajectories, N_boot, p].
    """
    n_trajectories, N, p = X_all.shape
    alphas = np.array([c0 / ((i + k0) ** step) for i in range(N)])
    theta_bar_boot_mean = np.zeros((n_trajectories, N_boot, 5, p))

    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        for traj in tqdm(range(n_trajectories), desc="Trajectories"):
            X = X_all[traj]
            y = y_all[traj]

            # разбиваем bootstrap в батчи
            batch_ids_list = [
                list(range(i, min(i + batch_size, N_boot)))
                for i in range(0, N_boot, batch_size)
            ]

            futures = {
                exe.submit(run_batch_trajectories, base_seed, batch_ids, traj, X, y, alphas, beta_a, beta_b): batch_ids
                for batch_ids in batch_ids_list
            }

            for fut in as_completed(futures):
                batch_results = fut.result()
                for traj_id, boot_id, theta_bars in batch_results:
                    theta_bar_boot_mean[traj_id, boot_id] = theta_bars
    np.savez_compressed(out_path,
                theta_bar_boot_mean=theta_bar_boot_mean,
                N_boot=N_boot,beta_a=beta_a, beta_b=beta_b)

    return out_path
