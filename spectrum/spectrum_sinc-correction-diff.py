import matplotlib.pyplot as plt
import numpy as np

# 1. ПАРАМЕТРЫ СИСТЕМЫ
df = 1e6  # Шаг поднесущих: 1 МГц
K = 8  # 8 информационных поднесущих: -4..-1 и +1..+4 МГц
N_fft = 9  # Размер БПФ: 8 информационных + 1 пустая нулевая (DC=0)
fs = N_fft * df  # Частота дискретизации I и Q ЦАП: 9 МГц
fs_mhz = fs / 1e6
oversample = 128  # Высокая передискретизация для точной имитации аналогового ZOH
fs_analog = fs * oversample
f_lo = 5000e6  # Гетеродин: 5000 МГц

# 2. Генерация 8 символов QPSK (16 бит)
rng = np.random.default_rng(42)
bits = rng.integers(0, 2, size=(K, 2))
qpsk_symbols = ((1 - 2 * bits[:, 0]) + 1j * (1 - 2 * bits[:, 1])) / np.sqrt(2)
subcarrier_freqs = np.array([-4, -3, -2, -1, 1, 2, 3, 4])

# 3. Заполнение спектра до и после inverse-sinc коррекции
spectrum_raw = np.zeros(N_fft, dtype=complex)
spectrum_comp = np.zeros(N_fft, dtype=complex)

for idx, k in enumerate(subcarrier_freqs):
  fft_bin = k if k > 0 else N_fft + k
  spectrum_raw[fft_bin] = qpsk_symbols[idx]
  inv_sinc_gain = 1.0 / np.sinc(k / fs_mhz)
  spectrum_comp[fft_bin] = qpsk_symbols[idx] * inv_sinc_gain

# Временной сигнал одного OFDM-символа и выход ZOH-ЦАП
x_iq_raw = np.fft.ifft(spectrum_raw) * N_fft
x_dac_raw = np.repeat(x_iq_raw, oversample)

x_iq_comp = np.fft.ifft(spectrum_comp) * N_fft
x_dac_comp = np.repeat(x_iq_comp, oversample)

N_sym = len(x_dac_comp)

# Точный БПФ по ровно одному периоду OFDM-символа (без утечки спектра)
freqs_dac = np.fft.fftshift(np.fft.fftfreq(N_sym, d=1.0 / fs_analog)) / 1e6
spec_dac_raw = np.fft.fftshift(np.abs(np.fft.fft(x_dac_raw))) / N_sym
spec_dac_comp = np.fft.fftshift(np.abs(np.fft.fft(x_dac_comp))) / N_sym

# Оставляем только ненулевые спектральные компоненты (гармоники с шагом 1 МГц)
active_mask = spec_dac_comp > 1e-4
f_tones = freqs_dac[active_mask]
amp_raw = spec_dac_raw[active_mask]
amp_comp = spec_dac_comp[active_mask]

# Динамический расчёт ключевых метрик для заголовков
k_edge = int(np.max(np.abs(subcarrier_freqs)))
max_gain = 1.0 / np.sinc(k_edge / fs_mhz)
max_gain_db = 20.0 * np.log10(max_gain)

idx_5mhz = np.argmin(np.abs(f_tones - 5.0))
peak_5_raw = amp_raw[idx_5mhz]
peak_5_comp = amp_comp[idx_5mhz]

f_lo_mhz = int(round(f_lo / 1e6))
rf_low_edge = f_lo_mhz - 5
rf_high_edge = f_lo_mhz + 5

# 4. Построение графиков
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(3, 1, figsize=(12, 11.8), constrained_layout=True)

# --- График 1: До IQ-ЦАП ---
ax = axes[0]
f_bins = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1.0 / fs)) / 1e6
s_bins_raw = np.fft.fftshift(np.abs(spectrum_raw))
s_bins_comp = np.fft.fftshift(np.abs(spectrum_comp))

ax.bar(
    f_bins - 0.14,
    s_bins_raw,
    width=0.25,
    color="gray",
    alpha=0.8,
    label="Без коррекции (амплитуда = 1.00)",
)
ax.bar(
    f_bins + 0.14,
    s_bins_comp,
    width=0.25,
    color="#2ca02c",
    alpha=0.9,
    label="С коррекцией 1/sinc(f/fs) в ПЛИС",
)

for idx, k in enumerate(subcarrier_freqs):
  b_str = f"{bits[idx, 0]}{bits[idx, 1]}"
  gain = 1.0 / np.sinc(k / fs_mhz)
  ax.text(
      k + 0.14,
      gain + 0.03,
      f"({b_str})\n×{gain:.2f}",
      ha="center",
      va="bottom",
      fontsize=8,
      fontweight="bold",
      color="#1b661b",
  )

ax.text(
    0,
    0.05,
    "k=0\n(DC=0)",
    ha="center",
    va="bottom",
    color="red",
    fontsize=8.5,
    fontweight="bold",
)
ax.axvline(-4.5, color="k", ls=":", lw=1.4, label="Границы Найквиста (±4.5 МГц)")
ax.axvline(4.5, color="k", ls=":", lw=1.4)
ax.set_xticks(range(-5, 6))
ax.set_title(
    f"1. До IQ-ЦАП (fs = {fs_mhz:g} МГц): предкоррекция поднимает края k=±{k_edge} в {max_gain:.2f} раза (+{max_gain_db:.1f} дБ)"
)
ax.set_xlabel("Частота в базовой полосе, МГц")
ax.set_ylabel("Цифровая амплитуда |X[k]|")
ax.set_xlim(-5.2, 5.2)
ax.set_ylim(0, 1.82)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper center", fontsize=8.5, ncol=3)

# --- График 2: Выход IQ-ЦАП (парные столбцы + обе огибающие) ---
ax = axes[1]
vis_mask = (f_tones >= -14) & (f_tones <= 14)
ft = f_tones[vis_mask]
ar = amp_raw[vis_mask]
ac = amp_comp[vis_mask]

ax.bar(
    ft - 0.14,
    ar,
    width=0.26,
    color="gray",
    alpha=0.75,
    label="Выход ЦАП без коррекции",
)
ax.bar(
    ft + 0.14,
    ac,
    width=0.26,
    color="#1f77b4",
    alpha=0.9,
    label="Выход ЦАП с коррекцией sinc",
)

# Теоретическая огибающая без коррекции
f_grid = np.linspace(-14.5, 14.5, 1000)
ax.plot(
    f_grid,
    np.abs(np.sinc(f_grid / fs_mhz)),
    "k--",
    lw=1.2,
    alpha=0.7,
    label="Огибающая до коррекции |sinc(f/fs)|",
)

# Теоретическая огибающая ПОСЛЕ коррекции в каждой зоне Найквиста
for n_zone in [-1, 0, 1]:
  f_z = np.linspace(n_zone * fs_mhz - 4.49, n_zone * fs_mhz + 4.49, 300)
  f0 = f_z - n_zone * fs_mhz
  env_comp = np.abs(np.sinc(f_z / fs_mhz) / np.sinc(f0 / fs_mhz))
  lbl = "Итоговая огибающая с коррекцией |f0 / f|" if n_zone == 0 else None
  ax.plot(f_z, env_comp, color="#d62728", lw=1.6, label=lbl)

for bnd in [-13.5, -4.5, 4.5, 13.5]:
  ax.axvline(bnd, color="k", ls=":", lw=1.2)

ax.axvspan(-13.5, -4.5, color="orange", alpha=0.08, label="Копия -1-й зоны (-fs)")
ax.axvspan(-4.5, 4.5, color="green", alpha=0.08, label="1-я зона Найквиста (основная)")
ax.axvspan(4.5, 13.5, color="purple", alpha=0.08, label="Копия +2-й зоны (+fs)")

# Подписываем ключевые изменения на краях зон (±4, ±5, ±13 МГц)
key_freqs = {4.0, 5.0, 13.0}
for freq_val, r_val, c_val in zip(ft, ar, ac):
  if round(abs(freq_val), 1) in key_freqs:
    ax.text(
        freq_val - 0.16,
        r_val + 0.02,
        f"{r_val:.2f}",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#444444",
    )
    ax.text(
        freq_val + 0.16,
        c_val + 0.02,
        f"{c_val:.2f}",
        ha="left",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color="#1f77b4",
    )

ax.set_xticks(range(-14, 15, 2))
ax.set_title(
    f"2. Выход IQ-ЦАП: коррекция меняет форму ВСЕХ зон (во 2-й зоне пик на +5 МГц вырос с {peak_5_raw:.2f} до {peak_5_comp:.2f})"
)
ax.set_xlabel("Частота в базовой полосе после ЦАП, МГц")
ax.set_ylabel("Амплитуда")
ax.set_xlim(-14.5, 14.5)
ax.set_ylim(0, 1.28)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=8, ncol=2)

# --- График 3: В эфире (5 ГГц) ---
ax = axes[2]
f_rf_tones = f_lo / 1e6 + ft
ax.bar(
    f_rf_tones - 0.14,
    ar,
    width=0.26,
    color="gray",
    alpha=0.75,
    label="В эфире без коррекции",
)
ax.bar(
    f_rf_tones + 0.14,
    ac,
    width=0.26,
    color="#d62728",
    alpha=0.9,
    label="В эфире с коррекцией sinc",
)

for bnd in [4986.5, 4995.5, 5004.5, 5013.5]:
  ax.axvline(bnd, color="k", ls=":", lw=1.2)
ax.axvline(5000, color="k", ls="--", lw=1, alpha=0.7, label="f_LO = 5000 МГц")

ax.axvspan(4986.5, 4995.5, color="orange", alpha=0.08, label="Нижняя реплика")
ax.axvspan(4995.5, 5004.5, color="green", alpha=0.08, label="Основной канал")
ax.axvspan(5004.5, 5013.5, color="purple", alpha=0.08, label="Верхняя реплика")

for freq_val, r_val, c_val in zip(ft, ar, ac):
  if round(abs(freq_val), 1) in key_freqs:
    rf_f = f_lo / 1e6 + freq_val
    ax.text(
        rf_f - 0.16,
        r_val + 0.02,
        f"{r_val:.2f}",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#444444",
    )
    ax.text(
        rf_f + 0.16,
        c_val + 0.02,
        f"{c_val:.2f}",
        ha="left",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color="#d62728",
    )

ax.set_xticks(range(4986, 5015, 2))
ax.set_title(
    f"3. В эфире ({f_lo / 1e9:g} ГГц): основной канал выровнен, но внеполосные пики на {rf_low_edge} и {rf_high_edge} МГц выросли до {peak_5_comp:.2f}"
)
ax.set_xlabel("Частота в радиоэфире, МГц")
ax.set_ylabel("Амплитуда")
ax.set_xlim(4985.5, 5014.5)
ax.set_ylim(0, 1.28)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=8, ncol=2)

plt.savefig("ofdm_iq_sinc_comparison_final.png", dpi=300)
plt.show()
