#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Монте-Карло верификация архитектуры частотного разнесения ПСИР (v8.2)
=====================================================================
Использует детерминированное управление фазой между интерливинговыми АЦП
для обратимого разделения спектральных реплик ЦАП с последующим MRC.

Ключевые моменты реализации:
1. Поподнесущая фазовая коррекция exp(-j*pi*f/fs) перед матрицей Адамара.
   Без неё линейный набег фазы от сдвига тактов dt=Ts/2 делает разделение
   валидным только на f=0. Коррекция убирает этот набег, оставляя строго
   детерминированную разность pi (знак "-") для зоны N=2.
2. Векторизация по поднесущим + адаптивный объём выборки (до 8e5 символов
   при высоких SNR) для достоверной оценки BER вплоть до 1e-5.
3. Baseline моделирует физически корректный случай одного АЦП со свёрткой
   зон N=1,2 со случайной фазой алиасинга -> деструктивная интерференция.
4. Идеальный MRC как верхняя граница (Perfect CSI, два независимых RF-тракта).

Автор: Рычков Е. Н., СФУ
Дата последней правки: сентябрь 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Параметры моделирования
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

N_SUB          = 64                          # кол-во поднесущих OFDM
SNR_DB_RANGE   = np.arange(6, 26, 2)        # 6..24 дБ, шаг 2
N_SYM_BASE     = 200_000                     # базовый объём символов
TARGET_BER     = [1e-2, 1e-3, 1e-4, 1e-5]   # целевые уровни для экстраполяции
VERBOSE        = True


def _log(msg):
    """Простой хелпер вывода — удобно глушить одним флагом."""
    if VERBOSE:
        print(msg)


# ---------------------------------------------------------------------------
# Генераторы и метрики
# ---------------------------------------------------------------------------
def qpsk_batch(n_sym, n_sub=N_SUB):
    """
    Массив QPSK размером (n_sym, n_sub), E[|s|^2]=1.
    Биты -> I/Q пары -> нормировка на sqrt(2).
    """
    bits = np.random.randint(0, 2, size=(n_sym, n_sub, 2))
    return ((2 * bits[..., 0] - 1) + 1j * (2 * bits[..., 1] - 1)) / np.sqrt(2.0)


def ber_hard(tx, rx):
    """
    BER при жёстком решении для QPSK.
    tx, rx -- массивы любой формы; считаются все биты (2 на символ).
    """
    tx_flat = tx.ravel()
    rx_flat = rx.ravel()
    dec_i = np.sign(rx_flat.real)
    dec_q = np.sign(rx_flat.imag)
    err = np.sum(dec_i != np.sign(tx_flat.real)) + \
          np.sum(dec_q != np.sign(tx_flat.imag))
    return err / (2.0 * tx_flat.size)


# ---------------------------------------------------------------------------
# Три рассматриваемые архитектуры
# ---------------------------------------------------------------------------
def arch_baseline(n_sym, snr_db):
    """
    Одна зона Найквиста, один АЦП.
    Реплики из зон N=1 и N=2 складываются со случайной фазой phi~U[0,2pi),
    что даёт эффективный одноветвлевой канал (d~1) с возможной деструктивной
    интерференцией при arg(h1) ~ arg(h2)+pi.
    """
    nv = 10.0 ** (-snr_db / 10.0)
    s  = qpsk_batch(n_sym)

    h1 = (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB)) / np.sqrt(2)
    h2 = (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB)) / np.sqrt(2)

    # случайная фаза алиасинга — вот почему d не растёт выше 1
    phi = np.random.uniform(0.0, 2.0*np.pi, size=(n_sym, N_SUB))
    h_eff_true = h1 + h2 * np.exp(1j * phi)

    y = h_eff_true * s
    n_data = np.sqrt(nv) * (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB))
    rx_noisy = y + n_data

    # LS-оценка канала по пилотам (шум оценки того же порядка, что и данные)
    n_pilot = np.sqrt(nv) * (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB))
    h_est = h_eff_true + n_pilot

    rx_eq = rx_noisy / (h_est + 1e-12)
    return s, rx_eq


def arch_ideal_mrc(n_sym, snr_db):
    """
    Верхняя граница: два полностью независимых RF-тракта, идеальное знание
    каналов h1, h2, когерентное MRC-сложение. Шум делится поровну.
    """
    nv = 10.0 ** (-snr_db / 10.0)
    s  = qpsk_batch(n_sym)

    h1 = (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB)) / np.sqrt(2)
    h2 = (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB)) / np.sqrt(2)

    n1 = np.sqrt(nv/2) * (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB))
    n2 = np.sqrt(nv/2) * (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB))

    den = np.abs(h1)**2 + np.abs(h2)**2 + 1e-12
    rx = (np.conj(h1)*(h1*s + n1) + np.conj(h2)*(h2*s + n2)) / den
    return s, rx


def arch_psir(n_sym, snr_db, nu_k, phase_corr):
    """
    Предлагаемая архитектура ПСИР.
    Два АЦП со сдвигом dt=Ts/2. Наблюдения после БПФ:
        Y1[k] = (h1+h2)*x[k] + n1
        Y2[k] = exp(j*pi*nu_k)*(h1-h2)*x[k] + n2
    Фазовая коррекция exp(-j*pi*nu_k) убирает частотно-зависимый набег,
    после чего система становится чистой матрицей Адамара 2x2 и обращение
    точно выделяет ветви alpha1*x и alpha2*x на всех поднесущих.
    """
    nv = 10.0 ** (-snr_db / 10.0)
    s  = qpsk_batch(n_sym)

    h1 = (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB)) / np.sqrt(2)
    h2 = (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB)) / np.sqrt(2)

    n1 = np.sqrt(nv/2) * (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB))
    n2 = np.sqrt(nv/2) * (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB))

    # ---- физика двухфазного интерливинга ----
    ep = np.exp(1j * np.pi * nu_k)[np.newaxis, :]   # (1, N_SUB) broadcast
    Y1 = (h1 + h2) * s + n1
    Y2 = ep * (h1 - h2) * s + n2

    # ---- КЛЮЧЕВОЙ ШАГ: поподнесущая коррекция фазы ----
    # Убираем линейный набег pi*f/fs, оставляем только знак "-" для зоны N=2.
    pc = phase_corr[np.newaxis, :]                  # (1, N_SUB)
    Y2c = Y2 * pc

    # ---- Разделение унитарным преобразованием Адамара ----
    # H2 = [[1,1],[1,-1]]/sqrt(2); здесь используем эквивалентную форму 0.5*(...)
    S1 = 0.5 * (Y1 + Y2c)                           # ~ h1*x
    S2 = 0.5 * (Y1 - Y2c)                           # ~ h2*x

    # ---- Оценка каналов по пилотам (LS) ----
    np1 = np.sqrt(nv/2) * (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB))
    np2 = np.sqrt(nv/2) * (np.random.randn(n_sym, N_SUB) + 1j*np.random.randn(n_sym, N_SUB))
    h1_hat = h1 + np1
    h2_hat = h2 + np2

    # ---- MRC комбинирование ----
    den = np.abs(h1_hat)**2 + np.abs(h2_hat)**2 + 1e-12
    out = (np.conj(h1_hat)*S1 + np.conj(h2_hat)*S2) / den
    return s, out


# ---------------------------------------------------------------------------
# Адаптивный прогон Монте-Карло
# ---------------------------------------------------------------------------
def sweep(fn, tag, extra_args=None):
    """
    Прогон по диапазону SNR с адаптивным числом символов.
    extra_args -- кортеж доп. позиционных аргументов для fn (если нужны).
    Возвращает массив BER той же длины, что и SNR_DB_RANGE.
    """
    bers = []
    _log(f"\n[{tag}] старт...")
    for snr in SNR_DB_RANGE:
        # чем выше SNR, тем больше нужно событий для стабильной оценки BER
        if snr >= 18:
            nsym = N_SYM_BASE * 4
        elif snr >= 14:
            nsym = N_SYM_BASE * 2
        else:
            nsym = N_SYM_BASE

        t0 = time.time()
        args = (nsym, snr) + tuple(extra_args or ())
        tx, rx = fn(*args)
        b = ber_hard(tx, rx)
        bers.append(b)
        print(f"  SNR={snr:2d} дБ | {nsym:>7d} сим. | BER={b:.3e} | "
              f"{time.time()-t0:.1f} с", flush=True)
    return np.array(bers)


# ---------------------------------------------------------------------------
# Статистическая обработка: порядок разнесения, экстраполяция, выигрыш
# ---------------------------------------------------------------------------
def fit_diversity(snr_arr, ber_arr, n_bits_total, label, targets=None):
    """
    Линейная регрессия log10(BER) vs log10(SNR_lin) в асимптотической области
    (SNR>=10 дБ, BER<=0.08). Наклон с минусом даёт оценку d.
    Дополнительно возвращает словарь экстраполированных SNR по целевым BER.
    """
    safe = np.maximum(ber_arr, 0.5 / max(n_bits_total, 1))
    mask = (snr_arr >= 10) & (safe <= 0.08)
    x_raw, y_raw = snr_arr[mask], safe[mask]

    if len(x_raw) < 3:
        _log(f"  [{label}] недостаточно точек для регрессии ({len(x_raw)})")
        return 0.0, np.inf, 0.0, 1.0, {}

    X = np.log10(10.0 ** (x_raw / 10.0))   # log10(SNR_linear)
    Y = np.log10(y_raw)                    # log10(BER)

    coef = np.polyfit(X, Y, deg=1)
    d_hat = -coef[0]                       # slope = -d => d = -slope

    Y_fit = np.polyval(coef, X)
    ss_res = float(np.sum((Y - Y_fit)**2))
    ss_tot = float(np.sum((Y - Y.mean())**2))
    r_sq = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0

    dof = max(len(X) - 2, 1)
    se_slope = np.sqrt(ss_res/dof) / np.sqrt(np.sum((X - X.mean())**2))
    p_val = 2.0 * stats.t.sf(abs(coef[0])/se_slope, df=dof) if se_slope > 0 else 1.0

    _log(f"  [{label:<18s}] d = {d_hat:.3f} +/- {se_slope:.3f} | "
         f"R^2 = {r_sq:.4f} | p = {p_val:.2e}")

    extrap = {}
    if targets:
        for tgt in targets:
            lg = np.log10(tgt)
            lin_log = (lg - coef[1]) / coef[0]
            extrap[tgt] = 10.0 * np.log10(10.0 ** lin_log)
    return d_hat, se_slope, r_sq, p_val, extrap


def snr_gap(ref_snr, ref_ber, prop_snr, prop_ber, targets):
    """
    Выигрыш по SNR: разница между required-SNR(ref) и required-SNR(prop)
    на каждом целевом уровне BER. Интерполяция с экстраполяцией за пределы.
    """
    g = {}
    fr = interp1d(ref_ber, ref_snr, kind='linear',
                  bounds_error=False, fill_value='extrapolate')
    fp = interp1d(prop_ber, prop_snr, kind='linear',
                  bounds_error=False, fill_value='extrapolate')
    for t in targets:
        g[t] = float(fr(t) - fp(t))
    return g


# ---------------------------------------------------------------------------
# Основной запуск
# ---------------------------------------------------------------------------
def main():
    banner = "=" * 80
    print(banner)
    print(" ВЕРИФИКАЦИЯ АРХИТЕКТУРЫ ПСИР (v8.2)")
    print(banner)
    print(f" Python {__import__('sys').version.split()[0]} | NumPy {np.__version__}")
    print(f" Запуск: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" K={N_SUB} подн. | SNR: {list(SNR_DB_RANGE)} дБ | "
          f"адаптив. объём до {N_SYM_BASE*4} симв.")
    print(banner)

    # предвычисляем векторы фазовой коррекции один раз
    nu_k      = np.linspace(0.0, 0.5, N_SUB, endpoint=True)
    phase_corr = np.exp(-1j * np.pi * nu_k)   # exp(-j*pi*f/fs)

    wall0 = time.time()

    ber_base  = sweep(arch_baseline,  "Baseline (1 ADC)")
    ber_ideal = sweep(arch_ideal_mrc, "Ideal MRC (предел)")
    ber_psir  = sweep(arch_psir,      "ПСИР (2 ADC, Hadamard+MRC)",
                      extra_args=(nu_k, phase_corr))

    print(f"\n Все прогоны завершены за {time.time()-wall0:.1f} с\n")

    # ---- статистика ----
    total_bits = N_SYM_BASE * N_SUB * 2   # консервативно: минимум выборки

    db, seb, r2b, pb, eb = fit_diversity(
        SNR_DB_RANGE, ber_base,  total_bits, "Baseline", TARGET_BER)
    di, sei, r2i, pi_, ei = fit_diversity(
        SNR_DB_RANGE, ber_ideal, total_bits, "Ideal MRC", TARGET_BER)
    dp, sep, r2p, pp, ep = fit_diversity(
        SNR_DB_RANGE, ber_psir,  total_bits, "ПСИР",      TARGET_BER)

    gains = snr_gap(SNR_DB_RANGE, ber_base, SNR_DB_RANGE, ber_psir, TARGET_BER)

    # ---- график ----
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogy(SNR_DB_RANGE, ber_base,  'b--o', lw=2, ms=4,
                label='Baseline (1 ADC, random-phase aliasing)')
    ax.semilogy(SNR_DB_RANGE, ber_psir,  'r-s',  lw=2, ms=4,
                label='ПСИР (2 ADC, Phase-Corrected Hadamard + MRC)')
    ax.semilogy(SNR_DB_RANGE, ber_ideal, 'g-.^', lw=2, ms=4,
                label='Ideal MRC (upper bound, Perfect CSI)')

    for lvl in TARGET_BER:
        ax.axhline(lvl, color='gray', ls=':', lw=0.5, alpha=0.35)
        ax.text(SNR_DB_RANGE[-1] + 0.3, lvl,
                f'$10^{{{int(np.log10(lvl))}}}$',
                fontsize=8, va='center', alpha=0.7)

    ax.set_xlabel('SNR, дБ')
    ax.set_ylabel('BER')
    ax.set_ylim(1e-6, 0.5)
    ax.set_xlim(SNR_DB_RANGE[0]-0.5, SNR_DB_RANGE[-1]+1.5)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.set_title(f'Архитектура ПСИР: символьная верификация (K={N_SUB})',
                 fontsize=12, pad=12)
    ax.legend(loc='lower left', fontsize=9)
    fig.tight_layout()
    fname = 'psid_v8_verify.png'
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    print(f" График сохранён: {fname}\n")

    # ---- итоговая таблица ----
    print(banner)
    print(" ИТОГИ")
    print(banner)
    hdr = f"{'Показатель':<32s} {'Baseline':>10s} {'ПСИР':>10s} {'Ideal':>10s}"
    print(hdr)
    print("-" * len(hdr))
    print(f"{'Порядок разнесения d':<32s} {db:>10.3f} {dp:>10.3f} {di:>10.3f}")
    print(f"{'Станд. ошибка d':<32s} {seb:>10.3f} {sep:>10.3f} {sei:>10.3f}")
    print(f"{'R^2 регрессии':<32s} {r2b:>10.4f} {r2p:>10.4f} {r2i:>10.4f}")
    print("-" * len(hdr))
    for tgt in TARGET_BER:
        gs = f"+{gains.get(tgt, np.nan):.2f} дБ" if np.isfinite(gains.get(tgt, np.nan)) else "N/A"
        print(f"{'SNR @ BER='+f'{tgt:.0e}':<32s} "
              f"{eb.get(tgt, np.nan):>7.2f}дБ {ep.get(tgt, np.nan):>7.2f}дБ "
              f"{ei.get(tgt, np.nan):>7.2f}дБ  ΔvsBase={gs}")

    # ---- критерии приёмки ----
    print("\n" + banner)
    print(" КРИТЕРИИ ВАЛИДАЦИИ")
    print(banner)
    tests = [
        (db < 1.2,               "Baseline d < 1.2"),
        (1.75 <= dp <= 2.15,     "ПСИР d ∈ [1.75, 2.15]"),
        (pp < 0.005,             "p-value < 0.005"),
        (r2p > 0.97,             "R^2 > 0.97"),
        (dp > db + 0.7,          "Δd > 0.7 относительно Baseline"),
        (sep < 0.10,             "std_err(d) < 0.10"),
    ]
    passed = all(ok for ok, _ in tests)
    for ok, desc in tests:
        mark = "[OK]" if ok else "[!!]"
        print(f"  {mark} {desc}")
    verdict = "ГОТОВО К ОТПРАВКЕ В ЖУРНАЛ" if passed else "НУЖНЫ ДОРАБОТКИ"
    print(f"\n >>> {verdict} <<<")
    print(banner)


if __name__ == "__main__":
    main()