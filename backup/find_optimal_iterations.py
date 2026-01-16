#!/usr/bin/env python3
"""
적절한 iteration 수와 burn_in을 찾는 종합 진단 도구

이 스크립트는 다음을 수행합니다:
1. Trace plots로 초기 burn-in 기간 식별
2. Running statistics로 수렴 확인
3. Gelman-Rubin R-hat (여러 체인 비교)
4. Effective Sample Size (ESS) 계산
5. MSE 수렴 확인
6. 최적 burn_in 및 iteration 수 추천
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, '.')
from Optimisation_and_High_Performance_Computing_project2025.team9_OHPC_submission.src.calibrate_parallel import X0, DATA_FILE, model
from plot_mc_histogram import sa_optimize_with_trace


def calculate_ess(samples, max_lag=None):
    """
    Effective Sample Size (ESS) 계산
    ESS = N / (1 + 2 * sum(autocorr))
    ESS가 높을수록 더 많은 독립 샘플을 의미
    """
    n_samples, n_params = samples.shape
    ess_values = np.zeros(n_params)
    
    if max_lag is None:
        max_lag = min(1000, n_samples // 10)
    
    for p in range(n_params):
        chain = samples[:, p]
        mean = np.mean(chain)
        chain_centered = chain - mean
        var = np.var(chain_centered)
        
        if var > 1e-10:
            autocorr = np.zeros(max_lag)
            for lag in range(1, max_lag):
                if lag >= n_samples:
                    break
                autocorr[lag-1] = np.mean(chain_centered[:-lag] * chain_centered[lag:]) / var
            
            # 첫 번째 음수 autocorrelation 이후는 무시
            first_neg = np.where(autocorr < 0)[0]
            if len(first_neg) > 0:
                cutoff = first_neg[0]
            else:
                cutoff = max_lag
            
            ess_values[p] = n_samples / (1 + 2 * np.sum(autocorr[:cutoff]))
        else:
            ess_values[p] = n_samples
    
    return ess_values


def calculate_gelman_rubin(chains):
    """
    Gelman-Rubin R-hat 통계 계산 (여러 체인 필요)
    R-hat < 1.01이면 수렴된 것으로 간주
    """
    n_chains, n_samples, n_params = chains.shape
    
    # Between-chain variance
    chain_means = np.mean(chains, axis=1)  # (n_chains, n_params)
    overall_mean = np.mean(chain_means, axis=0)  # (n_params,)
    B = n_samples / (n_chains - 1) * np.sum((chain_means - overall_mean)**2, axis=0)
    
    # Within-chain variance
    chain_vars = np.var(chains, axis=1, ddof=1)  # (n_chains, n_params)
    W = np.mean(chain_vars, axis=0)  # (n_params,)
    
    # Pooled variance
    var_pooled = (n_samples - 1) / n_samples * W + B / n_samples
    
    # R-hat
    rhat = np.sqrt(var_pooled / W)
    
    return rhat


def find_burn_in(samples, mse_values, window_size=1000):
    """
    적절한 burn_in 기간을 찾는 함수
    
    방법:
    1. Running mean이 안정화되는 지점 찾기
    2. MSE가 수렴하는 지점 찾기
    3. 여러 파라미터의 평균 변화율이 작아지는 지점 찾기
    """
    n_samples, n_params = samples.shape
    n_windows = n_samples // window_size
    
    # Running mean 계산
    running_means = np.zeros((n_windows, n_params))
    running_stds = np.zeros((n_windows, n_params))
    window_centers = np.zeros(n_windows)
    
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, n_samples)
        window_centers[i] = (start_idx + end_idx) / 2
        running_means[i] = np.mean(samples[start_idx:end_idx], axis=0)
        running_stds[i] = np.std(samples[start_idx:end_idx], axis=0)
    
    # Running mean의 변화율 계산
    mean_changes = np.zeros(n_windows - 1)
    for i in range(n_windows - 1):
        # 각 파라미터의 변화율의 평균
        changes = np.abs(running_means[i+1] - running_means[i]) / (np.abs(running_means[i]) + 1e-10)
        mean_changes[i] = np.mean(changes)
    
    # MSE의 변화율 계산
    mse_window = window_size
    n_mse_windows = len(mse_values) // mse_window
    mse_running = np.zeros(n_mse_windows)
    mse_changes = np.zeros(n_mse_windows - 1)
    
    for i in range(n_mse_windows):
        start_idx = i * mse_window
        end_idx = min((i + 1) * mse_window, len(mse_values))
        mse_running[i] = np.mean(mse_values[start_idx:end_idx])
    
    for i in range(n_mse_windows - 1):
        if mse_running[i] > 0:
            mse_changes[i] = abs(mse_running[i+1] - mse_running[i]) / mse_running[i]
        else:
            mse_changes[i] = 0
    
    # 안정화 기준: 변화율이 임계값 이하로 떨어지는 지점
    threshold = 0.01  # 1% 변화율
    stable_window = None
    
    # 마지막 50% 구간에서 안정화 지점 찾기
    search_start = max(0, len(mean_changes) // 2)
    for i in range(search_start, len(mean_changes)):
        if mean_changes[i] < threshold and mse_changes[min(i, len(mse_changes)-1)] < threshold:
            stable_window = i
            break
    
    if stable_window is not None:
        recommended_burn_in = int(window_centers[stable_window])
    else:
        # 안정화 지점을 찾지 못한 경우, 전체의 20%를 burn_in으로 추천
        recommended_burn_in = n_samples // 5
    
    return recommended_burn_in, window_centers, running_means, mean_changes, mse_changes


def assess_convergence(samples, mse_values, burn_in=0):
    """
    수렴 상태를 종합적으로 평가
    """
    n_samples, n_params = samples.shape
    
    # Burn-in 이후 샘플만 사용
    post_burnin_samples = samples[burn_in:]
    post_burnin_mse = mse_values[burn_in:]
    
    # ESS 계산
    ess_values = calculate_ess(post_burnin_samples)
    min_ess = np.min(ess_values)
    mean_ess = np.mean(ess_values)
    
    # MSE 수렴 확인
    n_post = len(post_burnin_mse)
    early_mse = np.mean(post_burnin_mse[:n_post//3])
    late_mse = np.mean(post_burnin_mse[2*n_post//3:])
    mse_change_ratio = abs(late_mse - early_mse) / (early_mse + 1e-10)
    
    # 파라미터 수렴 확인
    early_params = np.mean(post_burnin_samples[:n_post//3], axis=0)
    late_params = np.mean(post_burnin_samples[2*n_post//3:], axis=0)
    param_change_ratio = np.mean(np.abs(late_params - early_params) / (np.abs(early_params) + 1e-10))
    
    # 수렴 판단
    ess_sufficient = min_ess > len(post_burnin_samples) * 0.1  # 최소 10% ESS
    mse_stable = mse_change_ratio < 0.01  # MSE 변화 < 1%
    params_stable = param_change_ratio < 0.01  # 파라미터 변화 < 1%
    
    convergence_status = ess_sufficient and mse_stable and params_stable
    
    return {
        'converged': convergence_status,
        'min_ess': min_ess,
        'mean_ess': mean_ess,
        'ess_ratio': min_ess / len(post_burnin_samples),
        'mse_change_ratio': mse_change_ratio,
        'param_change_ratio': param_change_ratio,
        'ess_sufficient': ess_sufficient,
        'mse_stable': mse_stable,
        'params_stable': params_stable
    }


def plot_diagnostic_analysis(samples, mse_values, recommended_burn_in, 
                             window_centers, running_means, mean_changes, mse_changes,
                             output_file="optimal_iterations_analysis.png"):
    """
    종합 진단 플롯 생성
    """
    n_samples, n_params = samples.shape
    
    # 대표 파라미터 선택
    if n_params == 30:
        param_indices = [0, 1, 10, 11, 20, 21]
        param_names = ['T0_1', 'T0_2', 'Ts0', 'Ts1', 'Td0', 'Td1']
    else:
        param_indices = [0, 1, 2, 3, 4, 5]
        param_names = ['Ts1', 'Td1', 'Ts2', 'Td2', 'Ts3', 'Td3']
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Trace plots with burn_in 표시
    for i, (idx, name) in enumerate(zip(param_indices, param_names)):
        ax = plt.subplot(4, 3, i + 1)
        ax.plot(samples[:, idx], alpha=0.5, linewidth=0.3, color='blue')
        ax.axvline(recommended_burn_in, color='red', linestyle='--', 
                  linewidth=2, label=f'Recommended burn_in: {recommended_burn_in:,}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(f'{name}')
        ax.set_title(f'Trace Plot: {name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 2. Running mean 변화율
    ax = plt.subplot(4, 3, 7)
    ax.plot(window_centers[1:], mean_changes, linewidth=2, color='darkblue')
    ax.axhline(0.01, color='red', linestyle='--', linewidth=1.5, 
              label='Stability threshold (1%)')
    ax.axvline(recommended_burn_in, color='red', linestyle='--', 
              linewidth=2, label=f'Recommended burn_in: {recommended_burn_in:,}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Parameter Change Rate')
    ax.set_title('Parameter Stability (Running Mean Change)')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. MSE 변화율
    ax = plt.subplot(4, 3, 8)
    mse_window_centers = window_centers[:len(mse_changes)]
    ax.plot(mse_window_centers[1:], mse_changes, linewidth=2, color='darkgreen')
    ax.axhline(0.01, color='red', linestyle='--', linewidth=1.5, 
              label='Stability threshold (1%)')
    ax.axvline(recommended_burn_in, color='red', linestyle='--', 
              linewidth=2, label=f'Recommended burn_in: {recommended_burn_in:,}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE Change Rate')
    ax.set_title('MSE Stability')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. MSE 수렴
    ax = plt.subplot(4, 3, 9)
    ax.plot(mse_values, alpha=0.7, linewidth=0.5, color='purple')
    ax.axvline(recommended_burn_in, color='red', linestyle='--', 
              linewidth=2, label=f'Recommended burn_in: {recommended_burn_in:,}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE')
    ax.set_title('MSE Convergence')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 5. Burn-in 비교 (여러 burn_in 값에 대한 ESS)
    ax = plt.subplot(4, 3, 10)
    burn_in_candidates = [0, recommended_burn_in//2, recommended_burn_in, 
                          recommended_burn_in*2, recommended_burn_in*3]
    burn_in_candidates = [b for b in burn_in_candidates if b < n_samples]
    ess_at_burnin = []
    
    for bi in burn_in_candidates:
        post_samples = samples[bi:]
        ess = calculate_ess(post_samples)
        ess_at_burnin.append(np.min(ess))
    
    ax.bar(range(len(burn_in_candidates)), ess_at_burnin, alpha=0.7, color='coral')
    ax.set_xticks(range(len(burn_in_candidates)))
    ax.set_xticklabels([f'{bi:,}' for bi in burn_in_candidates], rotation=45)
    ax.set_xlabel('Burn-in')
    ax.set_ylabel('Min ESS')
    ax.set_title('ESS vs Burn-in')
    ax.axhline(len(samples) * 0.1, color='red', linestyle='--', 
              label='10% threshold', linewidth=1.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. 수렴 평가 요약
    ax = plt.subplot(4, 3, 11)
    ax.axis('off')
    
    # 여러 burn_in 값에 대한 수렴 평가
    convergence_summary = []
    for bi in burn_in_candidates:
        if bi < n_samples:
            result = assess_convergence(samples, mse_values, burn_in=bi)
            convergence_summary.append({
                'burn_in': bi,
                'converged': result['converged'],
                'ess_ratio': result['ess_ratio'],
                'mse_change': result['mse_change_ratio']
            })
    
    summary_text = f"""
    최적 Iteration 및 Burn-in 분석
    ==============================
    
    총 Iteration: {n_samples:,}
    
    추천 Burn-in: {recommended_burn_in:,} ({recommended_burn_in/n_samples*100:.1f}%)
    
    Burn-in별 수렴 평가:
    """
    for s in convergence_summary:
        status = "✓ 수렴" if s['converged'] else "✗ 미수렴"
        summary_text += f"\n  Burn-in {s['burn_in']:,}: {status}"
        summary_text += f"\n    ESS 비율: {s['ess_ratio']*100:.1f}%"
        summary_text += f"\n    MSE 변화: {s['mse_change']*100:.2f}%"
    
    # 최종 추천
    final_result = assess_convergence(samples, mse_values, burn_in=recommended_burn_in)
    summary_text += f"\n\n최종 추천 (burn_in={recommended_burn_in:,}):"
    summary_text += f"\n  수렴 상태: {'✓ 수렴됨' if final_result['converged'] else '✗ 미수렴'}"
    summary_text += f"\n  최소 ESS: {final_result['min_ess']:.0f}"
    summary_text += f"\n  ESS 비율: {final_result['ess_ratio']*100:.1f}%"
    summary_text += f"\n  MSE 변화: {final_result['mse_change_ratio']*100:.2f}%"
    
    if final_result['converged']:
        summary_text += f"\n\n✓ 현재 iteration 수({n_samples:,})는 충분합니다!"
    else:
        needed_iter = int(n_samples * (0.1 / final_result['ess_ratio']))
        summary_text += f"\n\n✗ 더 많은 iteration이 필요할 수 있습니다."
        summary_text += f"\n  추천: {needed_iter:,} iterations 이상"
    
    ax.text(0.05, 0.95, summary_text, fontsize=9, family='monospace',
           verticalalignment='top', transform=ax.transAxes)
    
    # 7. Running statistics (대표 파라미터)
    ax = plt.subplot(4, 3, 12)
    for i, idx in enumerate(param_indices[:3]):
        ax.plot(window_centers, running_means[:, idx], 
               label=param_names[i], linewidth=1.5, alpha=0.7)
    ax.axvline(recommended_burn_in, color='red', linestyle='--', 
              linewidth=2, label=f'Burn_in: {recommended_burn_in:,}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Running Mean')
    ax.set_title('Running Mean (Representative Parameters)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Optimal Iterations and Burn-in Analysis\n'
                f'Total: {n_samples:,} iterations | '
                f'Recommended burn_in: {recommended_burn_in:,}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved diagnostic analysis: {output_file}")
    plt.close()


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="적절한 iteration 수와 burn_in을 찾는 종합 진단 도구"
    )
    ap.add_argument("--T0", type=float, default=10.0, help="Initial temperature")
    ap.add_argument("--sigma", type=float, default=1e-5, help="Jump standard deviation")
    ap.add_argument("--n_iter", type=int, default=1000000, help="Total iterations")
    ap.add_argument("--outdir", type=str, default="results_burnin_tests",
                   help="Output directory")
    
    args = ap.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    
    # Load data
    print("="*60)
    print("적절한 Iteration 수와 Burn-in 찾기")
    print("="*60)
    print("\n데이터 로딩 중...")
    data = np.loadtxt(DATA_FILE, delimiter=',', skiprows=1)
    time_points = data[:, 0]
    data_points = data[:, 1]
    
    def mse(x):
        return float(np.mean((data_points - model(time_points, x)) ** 2))
    
    print(f"\nMonte Carlo 샘플링 실행 중...")
    print(f"T0={args.T0}, sigma={args.sigma}, n_iter={args.n_iter:,}\n")
    
    # Run sampling
    samples, mse_values = sa_optimize_with_trace(
        x0=X0,
        T0=args.T0,
        sigma=args.sigma,
        f=mse,
        n_iter=args.n_iter,
        seed=0
    )
    
    print(f"\n샘플링 완료!")
    
    # Burn-in 찾기
    print("\nBurn-in 분석 중...")
    recommended_burn_in, window_centers, running_means, mean_changes, mse_changes = \
        find_burn_in(samples, mse_values)
    
    print(f"추천 Burn-in: {recommended_burn_in:,} iterations ({recommended_burn_in/args.n_iter*100:.1f}%)")
    
    # 수렴 평가
    print("\n수렴 상태 평가 중...")
    convergence_result = assess_convergence(samples, mse_values, burn_in=recommended_burn_in)
    
    print("\n" + "="*60)
    print("수렴 평가 결과")
    print("="*60)
    print(f"수렴 상태: {'✓ 수렴됨' if convergence_result['converged'] else '✗ 미수렴'}")
    print(f"최소 ESS: {convergence_result['min_ess']:.0f}")
    print(f"평균 ESS: {convergence_result['mean_ess']:.0f}")
    print(f"ESS 비율: {convergence_result['ess_ratio']*100:.1f}%")
    print(f"MSE 변화율: {convergence_result['mse_change_ratio']*100:.2f}%")
    print(f"파라미터 변화율: {convergence_result['param_change_ratio']*100:.2f}%")
    
    if convergence_result['converged']:
        print(f"\n✓ 현재 iteration 수({args.n_iter:,})는 충분합니다!")
        print(f"✓ 추천 burn_in: {recommended_burn_in:,}")
    else:
        needed_iter = int(args.n_iter * (0.1 / convergence_result['ess_ratio']))
        print(f"\n✗ 더 많은 iteration이 필요할 수 있습니다.")
        print(f"  추천: {needed_iter:,} iterations 이상")
    
    # 진단 플롯 생성
    print("\n진단 플롯 생성 중...")
    output_file = outdir / f"optimal_iterations_analysis_iter{args.n_iter}.png"
    plot_diagnostic_analysis(samples, mse_values, recommended_burn_in,
                           window_centers, running_means, mean_changes, mse_changes,
                           str(output_file))
    
    print(f"\n분석 완료! 결과가 {outdir}/ 에 저장되었습니다.")
    print("="*60)


if __name__ == "__main__":
    main()
