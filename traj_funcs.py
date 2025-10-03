import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
import os

def run_single_trajectory(traj_id, N, p, sigma,  theta_true,
                          uniform_low, uniform_high,
                          base_seed, alphas):
    """
    Выполняет:
    - генерацию X (N x p) и y = X @ theta_true с seed = base_seed + traj_id
    - последовательный (batch=1) SGD с шагом gamma_n = gamma1 * n^{-alpha}
    - сохраняет историю raw theta_hat (размер (N+1,p)) и averaged theta_bar (N+1,p)
    Возвращает кортеж (traj_id, theta_hat_history, theta_bar_history)
    """
    seed = int(base_seed) + int(traj_id)
    rng = np.random.default_rng(seed)

    # Генерация фич и ответов (без шума => точное решение = theta_true)
    X = rng.uniform(low=uniform_low, high=uniform_high, size=(N, p))
    noise = rng.normal(0.0, sigma, size=N)
    y = X.dot(theta_true) + noise  # нет шума

    # Инициализация
    theta_hat = np.zeros(p, dtype=float)    # начальное значение (можно менять)
    theta_bar = np.zeros(p, dtype=float)
    theta_hat_history = np.zeros((N + 1, p), dtype=float)
    theta_bar_history = np.zeros((N + 1, p), dtype=float)

    theta_hat_history[0] = theta_hat.copy()
    theta_bar_history[0] = theta_bar.copy()

    # SGD: используя квадратичную loss L = (y - x^T theta)^2
    # градиент (для одного примера): -2 (y - x^T theta) x
    for n in range(1, N + 1):
        xn = X[n - 1]          # shape (p,)
        yn = y[n - 1]
        alpha_n = alphas[n-1]

        # градиент стохастический:
        residual = yn - xn.dot(theta_hat)
        grad = -2.0 * residual * xn   # ∇_θ (y - x^T θ)^2

        # обновление (batch size 1)
        theta_hat = theta_hat - alpha_n * grad

        # усреднение по Поляку-Руперту (обычное скользящее)
        theta_bar = ( (n - 1) * theta_bar + theta_hat ) / n

        theta_hat_history[n] = theta_hat.copy()
        theta_bar_history[n] = theta_bar.copy()

    return traj_id, theta_hat_history, theta_bar_history, X, y

def run_batch_trajectories(batch_ids, N, p, sigma, theta_true,
                           uniform_low, uniform_high,
                           base_seed, alphas):
    """
    Запуск пакета траекторий, чтобы снизить накладные расходы на pickle.
    batch_ids — список идентификаторов траекторий
    """
    results = []
    for traj_id in batch_ids:
        results.append(run_single_trajectory(traj_id, N, p, sigma,
                                             theta_true,
                                             uniform_low, uniform_high,
                                             base_seed, alphas))
    return results

def run_all_trajectories(n_trajectories=1024,
                         N=1024000,
                         p=10,
                         sigma=0.01,
                         uniform_low=-1.0,
                         uniform_high=1.0,
                         base_seed=12345,
                         c0 = 200.0,
                         k0 = 20000.0,
                         step = 0.65,
                         num_workers=None,
                         batch_size = 16,
                         out_path="sgd_trajectories.npz"):
    """
    Запуск всех траекторий параллельно и сохранение результата в out_path (.npz)
    Результат будет содержать:
      - theta_hat_histories: shape (n_trajectories, N+1, p)
      - theta_bar_histories: shape (n_trajectories, N+1, p)
      - theta_true: shape (p,)
    """
    alphas = np.zeros(N,dtype=float)
    for i in range(N):
        alphas[i] = c0/((i+k0)**step)
    mu = 1
    q = p // 2
    theta_true = np.concatenate([mu * np.ones(q), -(mu/10) * np.ones(p - q)])


    # выделяем массивы для результатов
    theta_hat_histories = np.zeros((n_trajectories, N + 1, p), dtype=float)
    theta_bar_histories = np.zeros((n_trajectories, N + 1, p), dtype=float)
    X_all = np.zeros((n_trajectories, N, p), dtype=float)
    y_all = np.zeros((n_trajectories, N), dtype=float)
    batch_ids_list = [list(range(i, min(i + batch_size, n_trajectories)))
                      for i in range(0, n_trajectories, batch_size)]
    
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = {exe.submit(run_batch_trajectories, batch_ids, N, p, sigma,
                              theta_true, uniform_low, uniform_high,
                              base_seed, alphas): batch_ids
                   for batch_ids in batch_ids_list}


        for fut in as_completed(futures):
            batch_results = fut.result()
            print("one_finish")
            for traj_id, th_hat_hist, th_bar_hist, X, y in batch_results:
                theta_hat_histories[traj_id] = th_hat_hist
                theta_bar_histories[traj_id] = th_bar_hist
                X_all[traj_id] = X
                y_all[traj_id] = y



    # Сохраняем все в .npz
    np.savez_compressed(out_path,
                        theta_hat_histories=theta_hat_histories,
                        theta_bar_histories=theta_bar_histories,
                        X_all=X_all,
                        y_all=y_all,
                        theta_true=theta_true,
                        N=N, p=p, sigma=sigma, n_trajectories=n_trajectories,
                        uniform_low=uniform_low, uniform_high=uniform_high,
                        base_seed=base_seed, c0=c0, k0=k0, gamma=step)
    return out_path