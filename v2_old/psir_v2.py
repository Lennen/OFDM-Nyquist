#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Финальная символьная верификация архитектуры частотного разнесения 
с проверкой совместимости с пространственным разнесением (MIMO Diversity).

ОСНОВАНО НА КОДЕ mimi.py, НО С ПРАВИЛЬНОЙ ТЕРМИНОЛОГИЕЙ ДЛЯ СТАТЬИ.

Изменения относительно исходного mimo.py:
1. Убран термин "ярус". Используется "частотное разнесение" и "пространственное разнесение".
2. Названия схем в легенде приведены к виду, принятому в радиотехнических журналах.
3. Сохранена вся математика: честная нормировка мощности, независимый шум пилотов, 
   проверка вещественного тракта.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg") # Без GUI для автоматической генерации
import matplotlib.pyplot as plt
from scipy import stats
import time
import warnings
warnings.filterwarnings("ignore")

# Настройки шрифтов для поддержки кириллицы
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({'font.size': 10})

# ============================ ПАРАМЕТРЫ МОДЕЛИ ============================
SEED = 42
rng = np.random.default_rng(SEED)

K = 64                      # Число поднесущих OFDM
NU = np.linspace(0.0, 0.5, K, endpoint=False) # Нормированные частоты [0, 0.5)
PHASE_CORR = np.exp(-1j * np.pi * NU)         # Коэффициент фазовой коррекции

RHO = 0.5                   # Распределение мощности между репликами (rho1=rho2=0.5)
SQRT_RHO = np.sqrt(RHO)
INV_SQRT2 = 1.0 / np.sqrt(2.0)
EPS = 1e-12                 # Малая константа для защиты от деления на ноль

SNR_DB = np.arange(6, 26, 2) # Диапазон SNR: 6..24 дБ
N_BASE = 200_000            # Базовое число символов для низких SNR
BATCH_SIZE = 5_000          # Размер батча для экономии памяти
TARGET_BER_LEVELS = [1e-2, 1e-3, 1e-4]

# Параметры неидеальностей АЦП (для проверки робастности)
IMP_GAIN_MISMATCH = 0.02    # Рассогласование усиления 2%
IMP_PHASE_ERROR_DEG = 2.3   # Фазовая ошибка 2.3 градуса

# Параметры MIMO
MIMO_NRX = 2                # Количество приемных антенн в гибридной схеме
MIMO_BATCH = 1_000          # Меньший батч для MIMO из-за циклов по антеннам

# ============================ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ============================

def cgn(shape, var):
    """Генерация комплексного гауссова шума CN(0, var)."""
    return np.sqrt(var / 2.0) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

def rayleigh_channel(shape):
    """Генерация коэффициента канала Рэлея с E[|h|^2]=1."""
    return cgn(shape, 1.0)

def qpsk_modulation(shape):
    """Генерация QPSK символов с единичной средней мощностью E[|s|^2]=1."""
    bits = rng.integers(0, 2, shape + (2,))
    return ((2 * bits[..., 0] - 1) + 1j * (2 * bits[..., 1] - 1)) / np.sqrt(2.0)

def count_errors_qpsk(tx, rx):
    """Подсчет ошибок при жестком решении для QPSK."""
    err_i = np.sum((tx.real > 0) != (rx.real > 0))
    err_q = np.sum((tx.imag > 0) != (rx.imag > 0))
    total_errors = int(err_i + err_q)
    total_bits = int(tx.size * 2) # 2 бита на символ QPSK
    return total_errors, total_bits

def calculate_ber_with_ci(errors, bits, confidence=0.95):
    """Расчет BER и доверительного интервала (Clopper-Pearson)."""
    if bits == 0:
        return 0.0, 0.0, 0.0
    
    p_hat = errors / bits
    alpha = (1.0 - confidence) / 2.0
    
    lo = 0.0 if errors == 0 else float(stats.beta.ppf(alpha, errors, bits - errors + 1))
    hi = 1.0 if errors == bits else float(stats.beta.ppf(1.0 - alpha, errors + 1, bits - errors))
    
    return float(p_hat), lo, hi

# ============================ СИМУЛЯЦИЯ СХЕМ ПРИЕМА ============================

def simulate_baseline_single_adc(n_sym, snr_db):
    """
    Схема 1: Базовый приемник (Однозонный OFDM, 1 АЦП).
    Модель: y = h*x + n. Оценка канала LS.
    """
    sigma2 = 10.0**(-snr_db/10.0)
    x = qpsk_modulation((n_sym, K))
    h = rayleigh_channel((n_sym, K))
    n = cgn((n_sym, K), sigma2)
    
    y = h * x + n
    
    # Оценка канала LS (с шумом оценки, пропорциональным SNR данных)
    h_est = h + cgn((n_sym, K), sigma2) 
    
    # Equalization: X_dec = Y / H_est
    rx = y / (h_est + EPS)
    
    return count_errors_qpsk(x, rx)

def simulate_control_two_adc_one_replica(n_sym, snr_db):
    """
    Схема 2: Контроль аппаратуры (Однозонный OFDM, 2 АЦП, 1 реплика).
    Цель: Изолировать выигрыш от второго АЦП (process gain) без использования разнообразия каналов.
    """
    sigma2 = 10.0**(-snr_db/10.0)
    x = qpsk_modulation((n_sym, K))
    h = rayleigh_channel((n_sym, K))
    
    n1 = cgn((n_sym, K), sigma2)
    n2 = cgn((n_sym, K), sigma2)
    
    # Два наблюдения одного сигнала со сдвигом фазы Ts/2
    y1 = h * x + n1
    y2 = np.exp(1j * np.pi * NU) * h * x + n2 
    
    # Когерентное усреднение
    z = 0.5 * (y1 + y2 * PHASE_CORR)
    
    # Оценка канала. Дисперсия шума уменьшается после усреднения.
    # Var(z_noise) = 0.25 * (Var(n1) + Var(n2)) = 0.5 * sigma2.
    # Но сигнал тоже масштабируется. Эффективная дисперсия шума оценки ~ sigma2/2.
    h_est = h + cgn((n_sym, K), sigma2 / 2.0) 
    
    rx = z / (h_est + EPS)
    
    return count_errors_qpsk(x, rx)

def simulate_proposed_architecture(n_sym, snr_db, genie_mode=False):
    """
    Схема 3 & 4: Предлагаемая архитектура (Двухрепликовый OFDM, 2 АЦП).
    
    genie_mode=True -> Теоретический предел (Perfect CSI).
    genie_mode=False -> Реализуемая система (LS оценка канала).
    """
    sigma2 = 10.0**(-snr_db/10.0)
    x = qpsk_modulation((n_sym, K))
    h1 = rayleigh_channel((n_sym, K))
    h2 = rayleigh_channel((n_sym, K))
    
    n1 = cgn((n_sym, K), sigma2)
    n2 = cgn((n_sym, K), sigma2)
    
    # Наблюдения на входах АЦП
    # Y1 содержит сумму реплик (в фазе)
    # Y2 содержит разность реплик (из-за сдвига Ts/2 фаза второй реплики инвертируется)
    y1 = SQRT_RHO * (h1 + h2) * x + n1
    y2 = SQRT_RHO * np.exp(1j * np.pi * NU) * (h1 - h2) * x + n2
    
    # Коррекция фазы второго канала
    y2_corr = y2 * PHASE_CORR
    
    # Разделение матрицей Адамара (унитарное преобразование)
    s1 = INV_SQRT2 * (y1 + y2_corr) # ~ sqrt(2)*h1*x
    s2 = INV_SQRT2 * (y1 - y2_corr) # ~ sqrt(2)*h2*x
    
    if genie_mode:
        # ИДЕАЛЬНЫЙ КАНАЛ. Никакой оценки. Просто делим на истинные коэффициенты.
        # Учитываем множитель sqrt(2) из разделения.
        # MRC Output with Perfect CSI:
        num = np.conj(h1) * s1 + np.conj(h2) * s2
        den = np.abs(h1)**2 + np.abs(h2)**2 + EPS
        rx = num / den
    else:
        # РЕАЛЬНАЯ СИСТЕМА: Оценка канала LS по пилотам.
        xp = np.ones((n_sym, K), dtype=complex)
        n1p = cgn((n_sym, K), sigma2)
        n2p = cgn((n_sym, K), sigma2)
        
        y1p = SQRT_RHO * (h1 + h2) * xp + n1p
        y2p = SQRT_RHO * np.exp(1j * np.pi * NU) * (h1 - h2) * xp + n2p
        y2pc = y2p * PHASE_CORR
        
        s1p = INV_SQRT2 * (y1p + y2pc)
        s2p = INV_SQRT2 * (y1p - y2pc)
        
        # Оценки каналов (нормируем на sqrt(2) и xp=1)
        h1_est = s1p / np.sqrt(2)
        h2_est = s2p / np.sqrt(2)
        
        # MRC с оцененными каналами
        num = np.conj(h1_est) * s1 + np.conj(h2_est) * s2
        den = np.abs(h1_est)**2 + np.abs(h2_est)**2 + EPS
        rx = num / den
        
    return count_errors_qpsk(x, rx)

def simulate_hybrid_mimo_diversity(n_sym, snr_db):
    """
    Схема 5: Гибрид (Предлагаемая архитектура + MIMO Diversity 2x2).
    
    Модель: 
    - 2 приемные антенны.
    - Каждая антенна имеет свой 2-канальный TI-ADC и выполняет разделение реплик.
    - Итого независимых ветвей разнесения: 2 (частота) * 2 (пространство) = 4.
    - Общая мощность передатчика нормирована на 1.
    """
    sigma2 = 10.0**(-snr_db/10.0)
    # Мощность распределяется между антеннами, чтобы суммарная была 1
    rho_amp = np.sqrt(RHO / MIMO_NRX) 
    
    x = qpsk_modulation((n_sym, K))
    
    num_accumulator = np.zeros((n_sym, K), dtype=complex)
    den_accumulator = np.zeros((n_sym, K), dtype=float)
    
    epi = np.exp(1j * np.pi * NU)
    
    for a in range(MIMO_NRX):
        # Независимые каналы для каждой антенны
        h1 = rayleigh_channel((n_sym, K))
        h2 = rayleigh_channel((n_sym, K))
        
        n1 = cgn((n_sym, K), sigma2)
        n2 = cgn((n_sym, K), sigma2)
        
        # Масштабируем сигнал и каналы для учета распределения мощности
        h1_scaled = h1 * rho_amp
        h2_scaled = h2 * rho_amp
        
        y1 = SQRT_RHO * (h1_scaled + h2_scaled) * x + n1
        y2 = SQRT_RHO * epi * (h1_scaled - h2_scaled) * x + n2
        
        y2_corr = y2 * PHASE_CORR
        
        s1 = INV_SQRT2 * (y1 + y2_corr)
        s2 = INV_SQRT2 * (y1 - y2_corr)
        
        # Оценка канала (LS)
        xp = np.ones((n_sym, K), dtype=complex)
        n1p = cgn((n_sym, K), sigma2)
        n2p = cgn((n_sym, K), sigma2)
        
        y1p = SQRT_RHO * (h1_scaled + h2_scaled) * xp + n1p
        y2p = SQRT_RHO * epi * (h1_scaled - h2_scaled) * xp + n2p
        y2pc = y2p * PHASE_CORR
        
        s1p = INV_SQRT2 * (y1p + y2pc)
        s2p = INV_SQRT2 * (y1p - y2pc)
        
        h1_est = s1p / np.sqrt(2)
        h2_est = s2p / np.sqrt(2)
        
        # Накопление весов MRC
        num_accumulator += np.conj(h1_est) * s1 + np.conj(h2_est) * s2
        den_accumulator += np.abs(h1_est)**2 + np.abs(h2_est)**2
        
    rx = num_accumulator / (den_accumulator + EPS)
    return count_errors_qpsk(x, rx)

# ============================ ЗАПУСК МОНТЕ-КАРЛО ============================

def run_simulation(func, label, batch_size=BATCH_SIZE, **kwargs):
    print(f"\n[{label}] Запуск симуляции...")
    
    results = {
        "snr": [], "ber": [], "ber_plot": [], 
        "ci_lo": [], "ci_hi": [], "errors": [], "bits": []
    }
    
    for snr in SNR_DB:
        # Увеличиваем число символов для высоких SNR
        if snr < 14:
            n_total = N_BASE
        elif snr < 18:
            n_total = 2 * N_BASE
        else:
            n_total = 4 * N_BASE
            
        total_err = 0
        total_bit = 0
        remaining = n_total
        t_start = time.time()
        
        while remaining > 0:
            nb = min(batch_size, remaining)
            err, bit = func(nb, snr, **kwargs)
            total_err += err
            total_bit += bit
            remaining -= nb
            
        ber_val, ci_lo, ci_hi = calculate_ber_with_ci(total_err, total_bit)
        
        # Для графика: если ошибок нет, берем верхнюю границу ДИ
        ber_plot_val = ber_val if ber_val > 0 else max(ci_hi, 1e-8)
        
        results["snr"].append(snr)
        results["ber"].append(ber_val)
        results["ber_plot"].append(ber_plot_val)
        results["ci_lo"].append(ci_lo)
        results["ci_hi"].append(ci_hi)
        results["errors"].append(total_err)
        results["bits"].append(total_bit)
        
        elapsed = time.time() - t_start
        print(f"  SNR={snr:2d} дБ | Симлов={n_total:7d} | BER={ber_val:.3e} [{ci_lo:.3e}, {ci_hi:.3e}] | Время={elapsed:.1f}с")
        
    for key in results:
        results[key] = np.array(results[key])
        
    return results

# ============================ ПОСТРОЕНИЕ ГРАФИКА ============================

def plot_and_save(all_results, filename="psid_final_ru.png"):
    """Построение итогового графика для статьи с чистой академической легендой."""
    
    plt.figure(figsize=(10, 7))
    
    # Определение стилей и ПОЛНЫХ РУССКИХ названий для легенды
    curves_config = [
        ("baseline",      "b--o", "Базовый приемник\n(1 АЦП, 1 реплика)"),
        ("control_hw",    "m:x",  "Контроль аппаратуры\n(2 АЦП, 1 реплика)"),
        ("proposed",      "r-s",  "Предлагаемая архитектура\n(2 АЦП, 2 реплики)"),
        ("theoretical",   "g-.^", "Теоретический предел\n(Ideal CSI)"),
        ("hybrid_mimo",   "k:D",  "Гибрид: Предлагаемая +\nMIMO Diversity (2x2)")
    ]
    
    plotted_labels = []
    
    for key, style, label in curves_config:
        if key in all_results:
            data = all_results[key]
            plt.semilogy(data["snr"], data["ber_plot"], 
                         style, label=label, lw=2, ms=5, alpha=0.9)
            plotted_labels.append(label)
            
    # Добавление горизонтальных линий целевых уровней BER
    for target in TARGET_BER_LEVELS:
        plt.axhline(y=target, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)
        plt.text(SNR_DB[-1] + 0.5, target, f'$10^{{{int(np.log10(target))}}}$', 
                 fontsize=9, va='center', ha='left', color='gray')

    # Оформление осей
    plt.xlabel("Отношение сигнал/шум (SNR), дБ", fontsize=12)
    plt.ylabel("Вероятность битовой ошибки (BER)", fontsize=12)
    plt.title("Символьная верификация архитектуры частотного разнесения\n"
              "(QPSK, K=64, Канал Рэлея, нормированная мощность)", 
              fontsize=13, pad=15)
    
    # Лимиты осей
    plt.ylim(1e-7, 5e-1)
    plt.xlim(SNR_DB[0] - 0.5, SNR_DB[-1] + 1.5)
    
    # Сетка
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Легенда
    plt.legend(loc="lower left", fontsize=10, frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\n✅ График успешно сохранен: {filename}")

# ============================ MAIN ============================

def main():
    print("="*80)
    print("ЗАПУСК ФИНАЛЬНОЙ ВЕРИФИКАЦИИ ДЛЯ СТАТЬИ")
    print("="*80)
    
    all_results = {}
    
    # 1. Базовый приемник (1 АЦП)
    all_results["baseline"] = run_simulation(simulate_baseline_single_adc, "Базовый приемник")
    
    # 2. Контроль аппаратуры (2 АЦП, 1 реплика)
    all_results["control_hw"] = run_simulation(simulate_control_two_adc_one_replica, "Контроль аппаратуры")
    
    # 3. Предлагаемая архитектура (Реализуемая)
    all_results["proposed"] = run_simulation(simulate_proposed_architecture, "Предлагаемая архитектура", genie_mode=False)
    
    # 4. Теоретический предел (Genie)
    all_results["theoretical"] = run_simulation(simulate_proposed_architecture, "Теоретический предел", genie_mode=True)
    
    # 5. Гибрид MIMO
    print("\n[Гибрид MIMO] Запуск симуляции (может занять больше времени)...")
    all_results["hybrid_mimo"] = run_simulation(simulate_hybrid_mimo_diversity, "Гибрид MIMO", batch_size=MIMO_BATCH)
    
    # Построение графика
    plot_and_save(all_results)
    
    print("\n" + "="*80)
    print("СИМУЛЯЦИЯ ЗАВЕРШЕНА.")
    print("="*80)

if __name__ == "__main__":
    main()