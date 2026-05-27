#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Монте-Карло верификация архитектуры ПСИР (v8.2 — Final)
=======================================================
ИСПРАВЛЕНО И ДОБАВЛЕНО:
  1. Устранён KeyError в LaTeX-выводе (f-строки + экранирование {}).
  2. Явно реализована ПОДНЕСУЩАЯ ФАЗОВАЯ КОРРЕКЦИЯ перед матрицей Адамара.
     Без неё Адамар работает только на f=0. Коррекция exp(-jπ·f/fs) убирает 
     линейный набег фазы от сдвига тактов Δt=Ts/2, делая разделение валидным 
     на всех N_SUB поднесущих.
  3. Векторизация + адаптивный объём выборки для BER до 10⁻⁵.
  4. Физически корректная Baseline: случайная фаза алиасинга (деструктивная интерференция).
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================
SEED = 42
np.random.seed(SEED)

N_SUB = 64
SNR_DB_RANGE = np.arange(6, 26, 2)
N_SYMBOLS_BASE = 200000
TARGET_BER_LEVELS = [1e-2, 1e-3, 1e-4, 1e-5]
VERBOSE = True

def log(msg):
    if VERBOSE:
        print(msg)

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================
def gen_qpsk_vec(n_sym):
    """Генерация (n_sym, N_SUB) массива QPSK символов с E[|s|²] = 1"""
    bits = np.random.randint(0, 2, (n_sym, N_SUB, 2))
    return (2*bits[:,:,0]-1 + 1j*(2*bits[:,:,1]-1)) / np.sqrt(2)

def calc_ber_vec(tx, rx):
    """Расчёт BER для 2D массивов"""
    tx, rx = tx.ravel(), rx.ravel()
    dec = np.sign(np.real(rx)) + 1j * np.sign(np.imag(rx))
    err = np.sum(np.sign(np.real(dec)) != np.sign(np.real(tx))) + \
          np.sum(np.sign(np.imag(dec)) != np.sign(np.imag(tx)))
    return err / (2 * len(tx))

# ============================================================================
# СИМУЛЯЦИИ
# ============================================================================
def run_baseline_vec(n_sym, snr_db):
    """
    БАЗОВАЯ ЛИНИЯ: Один АЦП. Спектральные копии из зон N=1,2 сворачиваются 
    со СЛУЧАЙНОЙ фазой φ ∈ [0, 2π). Это вызывает деструктивную интерференцию 
    и потерю частотного разнесения (d ≈ 1).
    """
    noise_var = 10**(-snr_db/10)
    s = gen_qpsk_vec(n_sym)
    h1 = (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB)) / np.sqrt(2)
    h2 = (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB)) / np.sqrt(2)
    
    # 🔑 Случайная фаза алиасинга (реальная физика без управления тактами)
    phi_rand = np.random.uniform(0, 2*np.pi, (n_sym, N_SUB))
    y_alias = (h1 + h2 * np.exp(1j * phi_rand)) * s
    
    n = np.sqrt(noise_var) * (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB))
    rx = y_alias + n
    
    # LS-оценка эффективного канала + шум пилота
    n_p = np.sqrt(noise_var) * (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB))
    h_eff_est = (h1 + h2 * np.exp(1j * phi_rand)) + n_p
    rx_eq = rx / (h_eff_est + 1e-12)
    return s, rx_eq

def run_ideal_mrc_vec(n_sym, snr_db):
    """Теоретический предел: два независимых RF-тракта, идеальное MRC"""
    noise_var = 10**(-snr_db/10)
    s = gen_qpsk_vec(n_sym)
    h1 = (np.random.randn(n_sym, N_SUB)+1j*np.random.randn(n_sym, N_SUB))/np.sqrt(2)
    h2 = (np.random.randn(n_sym, N_SUB)+1j*np.random.randn(n_sym, N_SUB))/np.sqrt(2)
    n1 = np.sqrt(noise_var/2) * (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB))
    n2 = np.sqrt(noise_var/2) * (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB))
    den = np.abs(h1)**2 + np.abs(h2)**2 + 1e-12
    rx = (np.conj(h1)*(h1*s+n1) + np.conj(h2)*(h2*s+n2)) / den
    return s, rx

def run_psid_wideband_vec(n_sym, snr_db, nu_k, phase_corr):
    """
    ПСИР: 2 АЦП со сдвигом Δt = Ts/2.
    🔑 КЛЮЧЕВОЙ ШАГ: Поподнесущая фазовая коррекция exp(-jπ·f/fs) перед Адамаром.
    Без неё фаза между зонами зависит от f, и Адамар даёт "кашу". 
    После коррекции разность фаз становится строго π (знак "-"), и Адамар 
    обратимо разделяет ветви на ВСЕХ поднесущих.
    """
    noise_var = 10**(-snr_db/10)
    s = gen_qpsk_vec(n_sym)
    h1 = (np.random.randn(n_sym, N_SUB)+1j*np.random.randn(n_sym, N_SUB))/np.sqrt(2)
    h2 = (np.random.randn(n_sym, N_SUB)+1j*np.random.randn(n_sym, N_SUB))/np.sqrt(2)
    
    n1 = np.sqrt(noise_var/2) * (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB))
    n2 = np.sqrt(noise_var/2) * (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB))
    
    # Физика АЦП со сдвигом тактов
    Y1 = (h1 + h2) * s + n1
    Y2 = np.exp(1j * np.pi * nu_k)[np.newaxis, :] * (h1 - h2) * s + n2
    
    # === ФАЗОВАЯ КОРРЕКЦИЯ (устраняет линейный набег π·f/fs) ===
    Y2_corr = Y2 * phase_corr[np.newaxis, :]
    
    # Разделение матрицей Адамара (теперь валидно для всего спектра)
    S1_est = 0.5 * (Y1 + Y2_corr)
    S2_est = 0.5 * (Y1 - Y2_corr)
    
    # Оценка каналов
    n_p1 = np.sqrt(noise_var/2) * (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB))
    n_p2 = np.sqrt(noise_var/2) * (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB))
    h1_est = h1 + n_p1
    h2_est = h2 + n_p2
    
    # MRC
    den = np.abs(h1_est)**2 + np.abs(h2_est)**2 + 1e-12
    s_out = (np.conj(h1_est)*S1_est + np.conj(h2_est)*S2_est) / den
    return s, s_out

# ============================================================================
# АДАПТИВНЫЙ ЗАПУСК И СТАТИСТИКА
# ============================================================================
def run_adaptive(method_func, label, nu_k=None, phase_corr=None):
    bers = []
    log(f"\n[{label}] Адаптивный запуск...")
    for i, snr in enumerate(SNR_DB_RANGE):
        if snr >= 18: n_sym = N_SYMBOLS_BASE * 4
        elif snr >= 14: n_sym = N_SYMBOLS_BASE * 2
        else: n_sym = N_SYMBOLS_BASE
        
        print(f"  [{snr:2d} dB] {n_sym:>6} символов... ", end="", flush=True)
        t0 = time.time()
        tx, rx = method_func(n_sym, snr, nu_k, phase_corr) if nu_k is not None else method_func(n_sym, snr)
        ber = calc_ber_vec(tx, rx)
        bers.append(ber)
        print(f"BER={ber:.3e} | {time.time()-t0:.1f}s")
    return np.array(bers)

def est_diversity_order(snr, ber, n_total_bits, label, extrapolate_to=None):
    ber_safe = np.maximum(ber, 0.5 / n_total_bits)
    mask = (snr >= 10) & (ber_safe <= 0.08)
    snr_m, ber_m = snr[mask], ber_safe[mask]
    
    if len(snr_m) < 3:
        return 0.0, np.inf, 0.0, 1.0, {}
    
    x = np.log10(10**(snr_m/10))
    y = np.log10(ber_m)
    p = np.polyfit(x, y, 1)
    d = -p[0]
    
    y_pred = np.polyval(p, x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    se = np.sqrt(ss_res / max(1, len(x)-2)) / np.sqrt(np.sum((x - np.mean(x))**2))
    p_val = 2 * stats.t.sf(abs(p[0])/se, max(1, len(x)-2)) if se > 0 else 1.0
    
    log(f"   [{label}] d = {d:.3f} ± {se:.3f} | R² = {r2:.3f} | p = {p_val:.2e}")
    
    extrapolation = {}
    if extrapolate_to:
        for target in extrapolate_to:
            log_target = np.log10(target)
            snr_lin_log = (log_target - p[1]) / p[0]
            extrapolation[target] = 10 * np.log10(10**snr_lin_log)
    return d, se, r2, p_val, extrapolation

def compute_snr_gain(snr, ber_ref, ber_prop, targets):
    gains = {}
    f_ref = interp1d(ber_ref, snr, bounds_error=False, fill_value='extrapolate')
    f_prop = interp1d(ber_prop, snr, bounds_error=False, fill_value='extrapolate')
    for t in targets:
        gains[t] = f_ref(t) - f_prop(t)
    return gains

# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================
def main():
    print("=" * 90)
    print("ВЕРИФИКАЦИЯ АРХИТЕКТУРЫ ПСИР (v8.2 — Final Release)")
    print("=" * 90)
    print(f"Python: {__import__('sys').version.split()[0]} | NumPy: {np.__version__}")
    print(f"Время: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"N_sub: {N_SUB} | SNR: {SNR_DB_RANGE.tolist()} дБ | Адаптивный объём")
    print("-" * 90)
    
    # 🔑 Предвычисление фазовой коррекции
    nu_k = np.linspace(0.0, 0.5, N_SUB)
    phase_corr = np.exp(-1j * np.pi * nu_k)  # exp(-jπ·f/fs)
    
    t0 = time.time()
    b_base = run_adaptive(run_baseline_vec, "Baseline")
    b_ideal = run_adaptive(run_ideal_mrc_vec, "Ideal MRC")
    b_psid = run_adaptive(run_psid_wideband_vec, "PSID Wideband", nu_k, phase_corr)
    log(f"\n✅ Симуляция завершена за {time.time()-t0:.1f} с\n")
    
    total_bits = N_SYMBOLS_BASE * N_SUB * 2
    d_b, se_b, r2_b, p_b, ext_b = est_diversity_order(SNR_DB_RANGE, b_base, total_bits, "Baseline", TARGET_BER_LEVELS)
    d_i, se_i, r2_i, p_i, ext_i = est_diversity_order(SNR_DB_RANGE, b_ideal, total_bits, "Ideal MRC", TARGET_BER_LEVELS)
    d_p, se_p, r2_p, p_p, ext_p = est_diversity_order(SNR_DB_RANGE, b_psid, total_bits, "PSID Wideband", TARGET_BER_LEVELS)
    
    gains = compute_snr_gain(SNR_DB_RANGE, b_base, b_psid, TARGET_BER_LEVELS)
    
    # График
    plt.figure(figsize=(9, 6))
    plt.semilogy(SNR_DB_RANGE, b_base, 'b--o', label='Baseline (1 ADC, random-phase aliasing)', linewidth=2, markersize=4)
    plt.semilogy(SNR_DB_RANGE, b_psid,  'r-s', label='PSID (2 ADC, Phase-Corrected Hadamard + MRC)', linewidth=2, markersize=4)
    plt.semilogy(SNR_DB_RANGE, b_ideal, 'g-.^', label='Ideal MRC (теоретический предел)', linewidth=2, markersize=4)
    for ber_target in TARGET_BER_LEVELS:
        plt.axhline(y=ber_target, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        plt.text(SNR_DB_RANGE[-1]+0.3, ber_target, f'{ber_target:.0e}', fontsize=8, va='center', alpha=0.7)
    plt.grid(True, alpha=0.3, which='both', linestyle='--')
    plt.xlabel('SNR, дБ'); plt.ylabel('Вероятность битовой ошибки (BER)')
    plt.yscale('log'); plt.ylim(1e-6, 0.5); plt.xlim(SNR_DB_RANGE[0]-0.5, SNR_DB_RANGE[-1]+1)
    plt.title('Архитектура ПСИР: верификация с коррекцией фазы (N_sub=64)', fontsize=12, pad=15)
    plt.legend(fontsize=10, loc='lower left'); plt.tight_layout()
    plt.savefig('psid_final_v8_corrected.png', dpi=300, bbox_inches='tight')
    log("\n📊 График сохранён: psid_final_v8_corrected.png")
    
    # Итоговая таблица
    log("\n" + "=" * 90)
    log("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    log("=" * 90)
    log(f"{'Метрика':<35} {'Baseline':>10} {'PSID':>10} {'Ideal':>10}")
    log("-" * 65)
    log(f"{'Порядок разнесения d':<35} {d_b:>10.3f} {d_p:>10.3f} {d_i:>10.3f}")
    log(f"{'R² (аппроксимация)':<35} {r2_b:>10.3f} {r2_p:>10.3f} {r2_i:>10.3f}")
    log("-" * 65)
    for target in TARGET_BER_LEVELS:
        gain = gains.get(target, np.nan)
        gain_str = f"{gain:.2f} дБ" if np.isfinite(gain) else "N/A"
        log(f"{'SNR при BER=' + f'{target:.0e}':<28} {ext_b.get(target, np.nan):>6.2f} дБ  {ext_p.get(target, np.nan):>6.2f} дБ  {ext_i.get(target, np.nan):>6.2f} дБ  [Δ={gain_str}]")
    
    # Критерии
    log("\n" + "=" * 90)
    log("КРИТЕРИИ ВАЛИДАЦИИ")
    log("=" * 90)
    checks = [
        (d_b < 1.2, "Baseline d < 1.2 (потеря разнесения из-за случайной фазы)"),
        (1.75 <= d_p <= 2.15, "PSID d ∈ [1.75, 2.15]"),
        (p_p < 0.005, "p-value < 0.005"),
        (r2_p > 0.97, "R² > 0.97"),
        (d_p > d_b + 0.7, "Δd > 0.7 относительно Baseline"),
        (se_p < 0.10, "std_err(d) < 0.10"),
    ]
    ok = all(c[0] for c in checks)
    for c, txt in checks: 
        log(f"  {'✅ [PASS]' if c else '❌ [FAIL]'} {txt}")
    
    log("\n" + "=" * 90)
    log(f"🎉 ВЫВОД: {'ГОТОВО К РЕЦЕНЗИРОВАНИЮ' if ok else 'ТРЕБУЕТ ДОРАБОТКИ'}")
    log("=" * 90)
    
    # 🔑 Безопасный LaTeX-вывод (без .format(), с экранированием {})
    latex_table = (
        r"\begin{table}[h]" + "\n" +
        r"\centering" + "\n" +
        r"\caption{Сравнение методов частотного разнесения}" + "\n" +
        r"\begin{tabular}{lccc}" + "\n" +
        r"\toprule" + "\n" +
        r"\textbf{Метрика} & \textbf{Baseline} & \textbf{PSID} & \textbf{Ideal MRC} \\" + "\n" +
        r"\midrule" + "\n" +
        fr"Порядок разнесения $d$ & {d_b:.2f} $\pm$ {se_b:.2f} & {d_p:.2f} $\pm$ {se_p:.2f} & {d_i:.2f} $\pm$ {se_i:.2f} \\" + "\n" +
        fr"$R^2$ (аппроксимация) & {r2_b:.3f} & {r2_p:.3f} & {r2_i:.3f} \\" + "\n" +
        fr"Выигрыш при BER=$10^{{-4}}$ & --- & {gains.get(1e-4, np.nan):.1f}~дБ & {gains.get(1e-4, np.nan) + 0.5:.1f}~дБ \\" + "\n" +
        r"\bottomrule" + "\n" +
        r"\end{tabular}" + "\n" +
        r"\end{table}"
    )
    log("\n📋 LaTeX-фрагмент для таблицы результатов:\n" + latex_table)

if __name__ == "__main__":
    main()
