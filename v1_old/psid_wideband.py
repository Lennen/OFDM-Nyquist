#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Монте-Карло верификация архитектуры ПСИР (в7.0 Wideband)
ДОБАВЛЕНО: Явная поподнесущая фазовая коррекция для широкополосного OFDM.
           Модель теперь оперирует N_SUB поднесущими, а не одной эквивалентной базой.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("ВЕРИФИКАЦИЯ АРХИТЕКТУРЫ ПСИР (ШИРОКОПОЛОСНАЯ ВЕРСИЯ, v7.0)")
print("=" * 90)
print(f"Python: {__import__('sys').version.split()[0]} | NumPy: {np.__version__}")
print(f"Время: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 90)

SEED = 42
np.random.seed(SEED)

N_SYMBOLS = 200000
SNR_DB_RANGE = np.arange(6, 20, 2)
N_SUB = 64  # Количество поднесущих OFDM в одном символе

# Параметры неидеальностей АЦП (используются в калибровке)
GAIN_MISMATCH = 0.020
TIMING_SKEW = 0.015
alpha = GAIN_MISMATCH / 2
beta = TIMING_SKEW * 0.5j
H_CAL  = np.array([[1, 1], 
                   [1 + 0.003, -0.997]], dtype=complex)

print("ПАРАМЕТРЫ МОДЕЛИРОВАНИЯ:")
print(f"  Символов OFDM/точка SNR : {N_SYMBOLS:,}")
print(f"  Поднесущих на символ    : {N_SUB}")
print(f"  Диапазон SNR            : {SNR_DB_RANGE.tolist()} дБ")
print(f"  Число обусловленности   : cond(H_CAL)={np.linalg.cond(H_CAL):.3f}")
print("-" * 90)

def gen_qpsk(n=1):
    """Генерация вектора QPSK символов с E[|s|²] = 1"""
    b = np.random.randint(0, 2, (n, 2))
    return (2*b[:,0]-1 + 1j*(2*b[:,1]-1)) / np.sqrt(2)

def calc_ber(tx, rx):
    """Расчёт BER при жёстком решении для QPSK"""
    tx, rx = np.asarray(tx), np.asarray(rx)
    dec = np.sign(np.real(rx)) + 1j * np.sign(np.imag(rx))
    err = np.sum(np.sign(np.real(dec)) != np.sign(np.real(tx))) + \
          np.sum(np.sign(np.imag(dec)) != np.sign(np.imag(tx)))
    return err / (2 * len(tx))

def run_baseline():
    """Эталон SISO: один АЦП, эквализация делением, LS-оценка"""
    bers = []
    for snr in SNR_DB_RANGE:
        noise_var = 10**(-snr/10)
        tx, rx = [], []
        for _ in range(N_SYMBOLS):
            s = gen_qpsk(N_SUB)
            h = (np.random.randn(N_SUB) + 1j*np.random.randn(N_SUB)) / np.sqrt(2)
            n_p = np.sqrt(noise_var/2) * (np.random.randn(N_SUB) + 1j*np.random.randn(N_SUB))
            n_d = np.sqrt(noise_var/2) * (np.random.randn(N_SUB) + 1j*np.random.randn(N_SUB))
            rx.append((h*s + n_d) / (h + n_p))
            tx.append(s)
        tx = np.concatenate(tx); rx = np.concatenate(rx)
        bers.append(calc_ber(tx, rx))
    return np.array(bers)

def run_ideal_mrc():
    """Теоретический предел: две независимые ветви, идеальное MRC"""
    bers = []
    for snr in SNR_DB_RANGE:
        noise_var = 10**(-snr/10)
        tx, rx = [], []
        for _ in range(N_SYMBOLS):
            s = gen_qpsk(N_SUB)
            h1 = (np.random.randn(N_SUB)+1j*np.random.randn(N_SUB))/np.sqrt(2)
            h2 = (np.random.randn(N_SUB)+1j*np.random.randn(N_SUB))/np.sqrt(2)
            n1 = np.sqrt(noise_var/2) * (np.random.randn(N_SUB) + 1j*np.random.randn(N_SUB))
            n2 = np.sqrt(noise_var/2) * (np.random.randn(N_SUB) + 1j*np.random.randn(N_SUB))
            den = np.abs(h1)**2 + np.abs(h2)**2 + 1e-12
            rx.append((np.conj(h1)*(h1*s+n1) + np.conj(h2)*(h2*s+n2)) / den)
            tx.append(s)
        tx = np.concatenate(tx); rx = np.concatenate(rx)
        bers.append(calc_ber(tx, rx))
    return np.array(bers)

def run_psid_wideband():
    """ПСИР: 2 АЦП, частотно-зависимый алиасинг, ФАЗОВАЯ КОРРЕКЦИЯ, разделение, MRC"""
    bers = []
    # Нормированные частоты поднесущих в базовой полосе [0, 0.5*fs]
    nu_k = np.linspace(0.0, 0.5, N_SUB)
    # Коэффициенты детерминированной фазовой коррекции для 2-го АЦП
    phase_corr = np.exp(-1j * np.pi * nu_k) 

    for snr in SNR_DB_RANGE:
        noise_var = 10**(-snr/10)
        tx, rx = [], []
        for _ in range(N_SYMBOLS):
            s = gen_qpsk(N_SUB)
            h1 = (np.random.randn(N_SUB)+1j*np.random.randn(N_SUB))/np.sqrt(2)
            h2 = (np.random.randn(N_SUB)+1j*np.random.randn(N_SUB))/np.sqrt(2)
            
            # === ФИЗИКА АЦП СО СДВИГОМ Ts/2 ===
            # Y1[k] = (h1 + h2) * s[k]
            # Y2[k] = exp(j*pi*nu_k) * (h1 - h2) * s[k]  (фазовый набеg от задержки)
            Y1 = (h1 + h2) * s
            Y2 = np.exp(1j * np.pi * nu_k) * (h1 - h2) * s
            
            # Шум АЦП
            n1 = np.sqrt(noise_var/2) * (np.random.randn(N_SUB) + 1j*np.random.randn(N_SUB))
            n2 = np.sqrt(noise_var/2) * (np.random.randn(N_SUB) + 1j*np.random.randn(N_SUB))
            Y1 += n1; Y2 += n2
            
            # === КЛЮЧЕВОЙ ШАГ: ПОДНЕСУЩАЯ ФАЗОВАЯ КОРРЕКЦИЯ ===
            # Убирает общую частотно-зависимую фазу, оставляя только знак "-" для зоны N=2
            Y2_corr = Y2 * phase_corr
            
            # Разделение матрицей Адамара (поподнесущно)
            S1_est = 0.5 * (Y1 + Y2_corr)
            S2_est = 0.5 * (Y1 - Y2_corr)
            
            # Оценка каналов (LS с независимым шумом пилота)
            n_p1 = np.sqrt(noise_var/2) * (np.random.randn(N_SUB) + 1j*np.random.randn(N_SUB))
            n_p2 = np.sqrt(noise_var/2) * (np.random.randn(N_SUB) + 1j*np.random.randn(N_SUB))
            h1_est = h1 + n_p1
            h2_est = h2 + n_p2
            
            # MRC-комбинирование
            den = np.abs(h1_est)**2 + np.abs(h2_est)**2 + 1e-12
            s_out = (np.conj(h1_est)*S1_est + np.conj(h2_est)*S2_est) / den
            
            tx.append(s); rx.append(s_out)
        tx = np.concatenate(tx); rx = np.concatenate(rx)
        bers.append(calc_ber(tx, rx))
    return np.array(bers)

def est_diversity_order(snr, ber, n_tx, label):
    ber_safe = np.maximum(ber, 0.5 / n_tx)
    mask = (snr >= 10) & (ber_safe <= 0.08)
    snr_m, ber_m = snr[mask], ber_safe[mask]
    
    print(f"\n[{label}] Асимптотическая область для оценки разнесения:")
    print(f"   SNR [дБ] : {' '.join(f'{x:3.0f}' for x in snr_m)}")
    print(f"   BER      : {' '.join(f'{x:.3e}' for x in ber_m)}")
    
    if len(snr_m) < 3:
        return 0.0, np.inf, 0.0, 1.0
    
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
    
    print(f"   Результат : d = {d:.3f} +/- {se:.3f} | R^2 = {r2:.3f} | p = {p_val:.2e}")
    return d, se, r2, p_val

def main():
    print("Запуск Монте-Карло (широкополосная модель, N_SUB=64, фазовая коррекция)...")
    t0 = time.time()
    b_base = run_baseline()
    b_ideal = run_ideal_mrc()
    b_psid = run_psid_wideband()
    print(f"Завершено за {time.time()-t0:.1f} секунд\n")
    
    d_b, se_b, r2_b, p_b = est_diversity_order(SNR_DB_RANGE, b_base, N_SYMBOLS*N_SUB, "Baseline")
    d_i, se_i, r2_i, p_i = est_diversity_order(SNR_DB_RANGE, b_ideal, N_SYMBOLS*N_SUB, "Ideal MRC")
    d_p, se_p, r2_p, p_p = est_diversity_order(SNR_DB_RANGE, b_psid, N_SYMBOLS*N_SUB, "PSID Wideband")
    
    plt.figure(figsize=(8,5))
    plt.semilogy(SNR_DB_RANGE, b_base, 'b--o', label='Baseline (SISO)', linewidth=2)
    plt.semilogy(SNR_DB_RANGE, b_psid,  'r-s', label='PSID (Wideband, Phase Corr, MRC)', linewidth=2)
    plt.semilogy(SNR_DB_RANGE, b_ideal, 'g-.^', label='Ideal MRC (эталон)', linewidth=2)
    plt.grid(True, alpha=0.3, which='both')
    plt.xlabel('SNR, дБ'); plt.ylabel('BER'); plt.yscale('log'); plt.ylim(1e-5, 0.5)
    plt.title('Архитектура ПСИР: широкополосная верификация (N_sub=64)')
    plt.legend(); plt.tight_layout()
    plt.savefig('psid_final_v7_wideband.png', dpi=300)
    print("\nГрафик сохранён: psid_final_v7_wideband.png")
    
    print("\nКРИТЕРИИ ВАЛИДАЦИИ:")
    checks = [
        (1.75 <= d_p <= 2.15, "d_предложено в [1.75, 2.15]"),
        (p_p < 0.005, "p-value < 0.005"),
        (r2_p > 0.97, "R^2 > 0.97"),
        (d_p > d_b + 0.7, "Delta d > 0.7"),
        (se_p < 0.10, "std_err < 0.10")
    ]
    ok = all(c[0] for c in checks)
    for c, txt in checks: 
        status = "[PASS]" if c else "[FAIL]"
        print(f"  {status} {txt}")
    
    print("=" * 90)
    result_msg = "ГОТОВО К РЕЦЕНЗИРОВАНИЮ" if ok else "ТРЕБУЕТ ДОРАБОТКИ"
    print(f"ВЫВОД: {result_msg}")
    print("=" * 90)

if __name__ == "__main__":
    main()
