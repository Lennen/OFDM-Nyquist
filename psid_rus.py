#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Монте-Карло верификация архитектуры фазо-сдвинутого интерливированного разнесения (ПСИР)
Эквивалентная базовополосная модель символьного уровня для архитектуры частотного разнесения.

Сопоставление переменных с физическими уровнями приёмника:
  - h1, h2 : Комплексные коэффициенты канала базовой полосы для зон Найквиста 1 и 2.
             Моделируются как независимые рэлеевские замирания, что справедливо при условии
             разноса зон fs >> когерентной полосы канала Bc.
  - H_TRUE : Матрица аналогового смешивания, возникающая при дискретизации двумя АЦП
             со сдвигом тактов на Ts/2. Диагональные элементы соответствуют идеальной
             привязке зон. Недиагональные элементы alpha (рассогласование усиления) и
             beta (фазовый сдвиг) моделируют типовые неидеальности интерливированных АЦП.
  - H_CAL  : Калиброванная матрица, хранящаяся в цифровой части (ПЗУ приёмника).
             Представляет результат заводской/фоновой калибровки структуры интерливированного АЦП.
             Используется для цифрового разделения зон.
  - n_adc  : Комплексный АБГШ, моделирующий тепловой и квантовый шум АЦП.
             Нормирован на единичную энергию символа (Es = 1). Применяется к аналоговой
             сумме после алиасинга.
  - v_sep  : Разделённые базовополосные потоки после обращения матрицы (H_CAL^-1 * y).
             Соответствует блоку цифрового разделения в ПЛИС/АСИК процессорах базовой полосы.
  - h1_est, h2_est : LS-оценки канала, полученные по мультиплексированным пилот-сигналам.
                     Независимый шум пилота/данных моделирует стандартное размещение пилотов в OFDM.
  - s_out  : Решающая статистика MRC-комбайнера. Реализует оптимальное линейное комбинирование
             при независимом шуме. Обеспечивает максимальный выигрыш разнесения, когда разделённые
             ветви некоррелированы.

Методология моделирования:
  - Символьный Монте-Карло по 2e5 реализаций на точку SNR.
  - Порядок разнесения d извлекается из асимптотического наклона BER посредством
    линейной регрессии в лог-лог шкале (SNR >= 10 дБ).
  - Соответствует стандартной практике IEEE/3GPP для начальной верификации архитектур
    приёмников с разнесением.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("ВЕРИФИКАЦИЯ АРХИТЕКТУРЫ ПСИР (ПУБЛИКАЦИОННАЯ ВЕРСИЯ, в6.1)")
print("=" * 90)
print(f"Python: {__import__('sys').version.split()[0]} | NumPy: {np.__version__}")
print(f"Время: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 90)

SEED = 42
np.random.seed(SEED)

N_SYMBOLS = 200000
SNR_DB_RANGE = np.arange(6, 20, 2)

# Параметры неидеальностей АЦП (типичные для высокоскоростных интерливированных преобразователей)
GAIN_MISMATCH = 0.020
TIMING_SKEW = 0.015

# Формирование истинной матрицы смешивания и калиброванной обратной
alpha = GAIN_MISMATCH / 2
beta = TIMING_SKEW * 0.5j
H_TRUE = np.array([[1, 1], 
                   [1 + alpha + beta, -1 + alpha - beta]], dtype=complex)
H_CAL  = np.array([[1, 1], 
                   [1 + 0.003, -0.997]], dtype=complex)
H_INV  = np.linalg.inv(H_CAL)

print("ПАРАМЕТРЫ МОДЕЛИРОВАНИЯ:")
print(f"  Символов на точку SNR : {N_SYMBOLS:,}")
print(f"  Диапазон SNR          : {SNR_DB_RANGE.tolist()} дБ")
print(f"  Число обусловленности H_TRUE : {np.linalg.cond(H_TRUE):.3f}")
print(f"  Число обусловленности H_CAL  : {np.linalg.cond(H_CAL):.3f}")
print(f"  Неидеальности АЦП     : Рассогласование усиления={GAIN_MISMATCH*100:.1f}%, фазовый сдвиг={TIMING_SKEW*100:.1f}%")
print("-" * 90)

def gen_qpsk():
    """Генерация символа QPSK с единичной средней мощностью (Es = 1)"""
    b = np.random.randint(0, 2, 2)
    return (1/np.sqrt(2)) * ((2*b[0]-1) + 1j*(2*b[1]-1))

def calc_ber(tx, rx):
    """Расчёт BER при жёстком решении для созвездия QPSK"""
    tx, rx = np.asarray(tx), np.asarray(rx)
    dec = np.sign(np.real(rx)) + 1j * np.sign(np.imag(rx))
    err = np.sum(np.sign(np.real(dec)) != np.sign(np.real(tx))) + \
          np.sum(np.sign(np.imag(dec)) != np.sign(np.imag(tx)))
    return err / (2 * len(tx))

def run_baseline():
    """Эталон SISO: один АЦП, зоны алиасируются в эффективный канал h = h1 + h2"""
    bers = []
    for snr in SNR_DB_RANGE:
        noise_var = 10**(-snr/10)
        tx, rx = [], []
        for _ in range(N_SYMBOLS):
            s = gen_qpsk()
            h = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)
            n_p = np.sqrt(noise_var/2) * (np.random.randn() + 1j*np.random.randn())
            n_d = np.sqrt(noise_var/2) * (np.random.randn() + 1j*np.random.randn())
            rx.append((h*s + n_d) / (h + n_p))
            tx.append(s)
        bers.append(calc_ber(np.array(tx), np.array(rx)))
    return np.array(bers)

def run_ideal_mrc():
    """Теоретический предел: две независимые РФ-цепи, идеальное MRC-комбинирование"""
    bers = []
    for snr in SNR_DB_RANGE:
        noise_var = 10**(-snr/10)
        tx, rx = [], []
        for _ in range(N_SYMBOLS):
            s = gen_qpsk()
            h1 = (np.random.randn()+1j*np.random.randn())/np.sqrt(2)
            h2 = (np.random.randn()+1j*np.random.randn())/np.sqrt(2)
            n1 = np.sqrt(noise_var/2) * (np.random.randn() + 1j*np.random.randn())
            n2 = np.sqrt(noise_var/2) * (np.random.randn() + 1j*np.random.randn())
            den = np.abs(h1)**2 + np.abs(h2)**2 + 1e-12
            rx.append((np.conj(h1)*(h1*s+n1) + np.conj(h2)*(h2*s+n2)) / den)
            tx.append(s)
        bers.append(calc_ber(np.array(tx), np.array(rx)))
    return np.array(bers)

def run_psid():
    """Предлагаемая архитектура: интерливированная дискретизация двумя АЦП, цифровое разделение, MRC"""
    bers = []
    for snr in SNR_DB_RANGE:
        noise_var = 10**(-snr/10)
        tx, rx = [], []
        for _ in range(N_SYMBOLS):
            s = gen_qpsk()
            h1 = (np.random.randn()+1j*np.random.randn())/np.sqrt(2)
            h2 = (np.random.randn()+1j*np.random.randn())/np.sqrt(2)
            v = np.array([h1*s, h2*s])
            
            # Аналоговый алиасинг на входах АЦП + тепловой/квантовый шум
            n_adc = np.sqrt(noise_var/2) * (np.random.randn(2) + 1j*np.random.randn(2))
            y = H_TRUE @ v + n_adc
            
            # Цифровое разделение зон через калиброванную обратную матрицу
            v_sep = H_INV @ y
            
            # Независимый шум пилота (моделирует мультиплексирование пилот/данные в OFDM)
            n_pilot = np.sqrt(noise_var/2) * (np.random.randn(2) + 1j*np.random.randn(2))
            v_p = H_INV @ (H_TRUE @ np.array([h1, h2])) + n_pilot
            h1_est, h2_est = v_p[0], v_p[1]
            
            # Веса MRC-комбайнера и решающая статистика
            den = np.abs(h1_est)**2 + np.abs(h2_est)**2 + 1e-12
            s_out = (np.conj(h1_est)*v_sep[0] + np.conj(h2_est)*v_sep[1]) / den
            
            tx.append(s)
            rx.append(s_out)
        bers.append(calc_ber(np.array(tx), np.array(rx)))
    return np.array(bers)

def est_diversity_order(snr, ber, n_tx, label):
    """Извлечение порядка разнесения d из асимптотического наклона BER (лог-лог регрессия)"""
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
    print("Запуск Монте-Карло моделирования (мгновенная LS-оценка, независимый шум)...")
    t0 = time.time()
    b_base = run_baseline()
    b_ideal = run_ideal_mrc()
    b_psid = run_psid()
    print(f"Завершено за {time.time()-t0:.1f} секунд\n")
    
    d_b, se_b, r2_b, p_b = est_diversity_order(SNR_DB_RANGE, b_base, N_SYMBOLS, "Baseline")
    d_i, se_i, r2_i, p_i = est_diversity_order(SNR_DB_RANGE, b_ideal, N_SYMBOLS, "Ideal MRC")
    d_p, se_p, r2_p, p_p = est_diversity_order(SNR_DB_RANGE, b_psid, N_SYMBOLS, "PSID")
    
    plt.figure(figsize=(8,5))
    plt.semilogy(SNR_DB_RANGE, b_base, 'b--o', label='Baseline (SISO)', linewidth=2)
    plt.semilogy(SNR_DB_RANGE, b_psid,  'r-s', label='PSID (2 АЦП, H_CAL, MRC)', linewidth=2)
    plt.semilogy(SNR_DB_RANGE, b_ideal, 'g-.^', label='Ideal MRC (эталон)', linewidth=2)
    plt.grid(True, alpha=0.3, which='both')
    plt.xlabel('SNR, дБ'); plt.ylabel('BER'); plt.yscale('log'); plt.ylim(1e-5, 0.5)
    plt.title('Архитектура ПСИР: верификация порядка частотного разнесения')
    plt.legend(); plt.tight_layout()
    plt.savefig('psid_final_v6.1.png', dpi=300)
    print("\nГрафик сохранён: psid_final_v6.1.png")
    
    print("\nКРИТЕРИИ ВАЛИДАЦИИ:")
    checks = [
        (1.75 <= d_p <= 2.15, "d_предложено в [1.75, 2.15] (учтены неидеальности)"),
        (p_p < 0.005, "p-value < 0.005"),
        (r2_p > 0.97, "R^2 > 0.97"),
        (d_p > d_b + 0.7, "Delta d > 0.7 (улучшение относительно SISO)"),
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
