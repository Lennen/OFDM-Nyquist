import matplotlib.pyplot as plt
import numpy as np

# 1. ПРЕДЕЛЬНЫЕ ПАРАМЕТРЫ СИСТЕМЫ (комплексный IQ-модем без лишних нулей)
df = 1e6  # Шаг поднесущих: 1 МГц
K = 8  # Всего 8 информационных поднесущих: -4..-1 и +1..+4 МГц
N_fft = 9  # Размер БПФ: 8 информационных + 1 пустая нулевая (DC=0)
fs = (
    N_fft * df
)  # Минимальная частота дискретизации I и Q ЦАП: 9 МГц (зона Найквиста: -4.5..+4.5 МГц)
oversample = 32  # Передискретизация для имитации выхода I/Q ЦАП (ZOH)
fs_analog = fs * oversample
f_lo = 5000e6  # Квадратурный гетеродин: 5 ГГц (5000 МГц)

# 2. Генерация 8 независимых символов QPSK (16 бит информации)
rng = np.random.default_rng(42)
bits = rng.integers(0, 2, size=(K, 2))
qpsk_symbols = ((1 - 2 * bits[:, 0]) + 1j * (1 - 2 * bits[:, 1])) / np.sqrt(2)

# Частотные индексы поднесущих в реальном OFDM (симметрично вокруг DC=0, сам DC=0 пустой)
subcarrier_freqs = [-4, -3, -2, -1, 1, 2, 3, 4]

# 3. Заполнение комплексного спектра перед IFFT (в стандартном порядке np.fft: 0..+4, -4..-1)
spectrum_digital = np.zeros(N_fft, dtype=complex)
for idx, k in enumerate(subcarrier_freqs):
  fft_bin = k if k > 0 else N_fft + k
  spectrum_digital[fft_bin] = qpsk_symbols[idx]

# Комплексный временной сигнал I(t) + j*Q(t) и выход пары ЦАП (ZOH)
x_iq = np.fft.ifft(spectrum_digital) * N_fft
x_dac_stream = np.tile(np.repeat(x_iq, oversample), 40)
N_total = len(x_dac_stream)

# Двусторонний БПФ-анализ комплексного сигнала после ЦАП
freqs_dac = np.fft.fftshift(np.fft.fftfreq(N_total, d=1.0 / fs_analog))
spec_dac = np.fft.fftshift(np.abs(np.fft.fft(x_dac_stream))) / N_total

# 4. Построение трёх графиков
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), constrained_layout=True)

# --- График 1: До IQ-ЦАП (предельная упаковка в 1-й зоне -4.5..+4.5 МГц) ---
ax = axes[0]
f_bins = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1.0 / fs)) / 1e6
s_bins = np.fft.fftshift(np.abs(spectrum_digital))

ax.stem(f_bins, s_bins, linefmt="C0-", markerfmt="C0o", basefmt="k-")
ax.stem([0], [0], linefmt="r--", markerfmt="rx", basefmt="k-")
ax.text(
    0,
    0.08,
    "k=0\n(DC=0)",
    ha="center",
    va="bottom",
    color="red",
    fontsize=8.5,
    fontweight="bold",
)

for idx, k in enumerate(subcarrier_freqs):
  b_str = f"{bits[idx, 0]}{bits[idx, 1]}"
  color_txt = "#d95f02" if k < 0 else "#1f77b4"
  ax.text(
      k,
      1.05,
      f"k={k}\n({b_str})",
      ha="center",
      va="bottom",
      fontsize=8.5,
      fontweight="bold",
      color=color_txt,
  )

ax.axvspan(
    -4.5,
    -0.5,
    color="orange",
    alpha=0.12,
    label="Нижняя боковая полоса (-4..-1 МГц, 4 поднесущие)",
)
ax.axvspan(
    0.5,
    4.5,
    color="green",
    alpha=0.12,
    label="Верхняя боковая полоса (+1..+4 МГц, 4 поднесущие)",
)
ax.axvline(
    -4.5, color="k", ls=":", lw=1.5, label="Границы Найквиста (±fs/2 = ±4.5 МГц)"
)
ax.axvline(4.5, color="k", ls=":", lw=1.5)
ax.axvline(0, color="red", ls="--", lw=1, alpha=0.7)

ax.set_xticks(range(-5, 6))
ax.set_title(
    "1. Предельный случай (fs = 9 МГц): 8 поднесущих + DC занимают 100% полосы"
    " от -4.5 до +4.5 МГц"
)
ax.set_xlabel("Частота в базовой полосе, МГц")
ax.set_ylabel("Амплитуда |X[k]|")
ax.set_xlim(-5.2, 5.2)
ax.set_ylim(0, 1.38)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=8.5)

# --- График 2: Выход IQ-ЦАП (периодическое размножение стык-в-стык) ---
ax = axes[1]
mask1 = (freqs_dac >= -14.5e6) & (freqs_dac <= 14.5e6)
f1 = freqs_dac[mask1] / 1e6
s1 = spec_dac[mask1]
fs_mhz = fs / 1e6

ax.plot(
    f1,
    s1,
    color="#1f77b4",
    lw=1.3,
    label="Спектр комплексного сигнала I+jQ после ЦАП (fs = 9 МГц)",
)
ax.plot(
    f1,
    np.abs(np.sinc(f1 / fs_mhz)),
    "r--",
    lw=1.2,
    label="Огибающая ЦАП |sinc(f / 9 МГц)|",
)

for bnd in [-13.5, -4.5, 4.5, 13.5]:
  ax.axvline(bnd, color="k", ls=":", lw=1.2)
ax.axvline(0, color="red", ls="--", lw=0.9, alpha=0.6)

ax.axvspan(
    -13.5,
    -4.5,
    color="orange",
    alpha=0.10,
    label="Копия -1-й зоны (-13.5..-4.5 МГц, сдвиг -fs)",
)
ax.axvspan(
    -4.5,
    4.5,
    color="green",
    alpha=0.10,
    label="1-я зона Найквиста (-4.5..+4.5 МГц, основная)",
)
ax.axvspan(
    4.5,
    13.5,
    color="purple",
    alpha=0.10,
    label="Копия +2-й зоны (+4.5..+13.5 МГц, сдвиг +fs)",
)

for k in subcarrier_freqs:
  # Основная зона: f = k
  a_z1 = np.abs(np.sinc(k / fs_mhz))
  ax.text(
      k,
      a_z1 + 0.03,
      f"{k}",
      ha="center",
      va="bottom",
      fontsize=8,
      color="darkgreen",
      fontweight="bold",
  )
  # Копия +fs: f = 9 + k
  f_z2 = fs_mhz + k
  a_z2 = np.abs(np.sinc(f_z2 / fs_mhz))
  ax.text(
      f_z2,
      a_z2 + 0.03,
      f"{k}",
      ha="center",
      va="bottom",
      fontsize=8,
      color="indigo",
      fontweight="bold",
  )
  # Копия -fs: f = -9 + k
  f_zm1 = -fs_mhz + k
  a_zm1 = np.abs(np.sinc(f_zm1 / fs_mhz))
  ax.text(
      f_zm1,
      a_zm1 + 0.03,
      f"{k}",
      ha="center",
      va="bottom",
      fontsize=8,
      color="#b35900",
      fontweight="bold",
  )

ax.set_xticks(range(-14, 15, 2))
ax.set_title(
    "2. На выходе IQ-ЦАП при fs = 9 МГц: копии идут СТЫК-В-СТЫК через каждые"
    " 9 МГц без инверсии"
)
ax.set_xlabel("Частота в базовой полосе после ЦАП, МГц")
ax.set_ylabel("Амплитуда")
ax.set_xlim(-14.5, 14.5)
ax.set_ylim(0, 1.25)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=8, ncol=2)

# --- График 3: После квадратурного модулятора 5 ГГц (в эфире) ---
ax = axes[2]
freqs_rf = (f_lo + freqs_dac) / 1e6
mask2 = (freqs_rf >= 4985.5) & (freqs_rf <= 5014.5)

ax.plot(
    freqs_rf[mask2],
    spec_dac[mask2],
    color="#d62728",
    lw=1.3,
    label="Сигнал в эфире после IQ-модулятора (5 ГГц)",
)
ax.axvline(
    5000,
    color="k",
    ls="--",
    lw=1,
    alpha=0.7,
    label="Несущая гетеродина f_LO = 5000 МГц (в центре пустого DC)",
)

ax.axvspan(
    4986.5,
    4995.5,
    color="orange",
    alpha=0.10,
    label="Нижняя реплика (4986.5..4995.5 МГц)",
)
ax.axvspan(
    4995.5,
    5004.5,
    color="green",
    alpha=0.10,
    label="Основной канал (4995.5..5004.5 МГц)",
)
ax.axvspan(
    5004.5,
    5013.5,
    color="purple",
    alpha=0.10,
    label="Верхняя реплика (5004.5..5013.5 МГц)",
)

for k in subcarrier_freqs:
  a_z1 = np.abs(np.sinc(k / fs_mhz))
  ax.text(
      5000 + k,
      a_z1 + 0.03,
      f"{k}",
      ha="center",
      va="bottom",
      fontsize=8,
      color="darkgreen",
      fontweight="bold",
  )
  f_z2 = fs_mhz + k
  a_z2 = np.abs(np.sinc(f_z2 / fs_mhz))
  ax.text(
      5000 + f_z2,
      a_z2 + 0.03,
      f"{k}",
      ha="center",
      va="bottom",
      fontsize=8,
      color="indigo",
      fontweight="bold",
  )
  f_zm1 = -fs_mhz + k
  a_zm1 = np.abs(np.sinc(f_zm1 / fs_mhz))
  ax.text(
      5000 + f_zm1,
      a_zm1 + 0.03,
      f"{k}",
      ha="center",
      va="bottom",
      fontsize=8,
      color="#b35900",
      fontweight="bold",
  )

ax.set_xticks(range(4986, 5015, 2))
ax.set_title(
    "3. В эфире при предельной fs = 9 МГц: поднесущая +4 (5004 МГц) вплотную"
    " граничит с -4 (5005 МГц) следующей копии"
)
ax.set_xlabel("Частота в радиоэфире, МГц")
ax.set_ylabel("Амплитуда")
ax.set_xlim(4985.5, 5014.5)
ax.set_ylim(0, 1.25)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=8, ncol=2)

plt.savefig("ofdm_iq_limit_9mhz.png", dpi=300)
plt.show()
