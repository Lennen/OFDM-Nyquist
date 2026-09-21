#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte-Carlo Verification of Phase-Shifted Interleaved Diversity (PSID)
Baseband-equivalent symbol-level model for frequency diversity architecture.

Physical layer mapping:
  - h1, h2 : Complex baseband channel coefficients for Nyquist Zone 1 and Zone 2.
             Modeled as i.i.d. Rayleigh fading, valid when zone spacing fs >> coherence bandwidth Bc.
  - H_TRUE : Analog mixing matrix induced by dual-ADC sampling with offset Ts/2.
             Diagonal terms represent ideal zone mapping. Off-diagonal terms alpha (gain mismatch)
             and beta (timing skew) model typical time-interleaved ADC impairments.
  - H_CAL  : Digitally stored calibration matrix (firmware). Represents foreground/factory
             calibration of the interleaved ADC structure. Used for digital zone separation.
  - n_adc  : Complex AWGN modeling thermal and quantization noise of the ADCs.
             Normalized to unit symbol energy (Es = 1). Applied to the aliased analog sum.
  - v_sep  : Digitally separated baseband streams after matrix inversion (H_CAL^-1 * y).
             Corresponds to the DSP separation block in FPGA/ASIC baseband processors.
  - h1_est, h2_est : LS channel estimates derived from multiplexed pilot symbols.
                     Independent pilot/data noise models standard OFDM pilot placement.
  - s_out  : MRC decision variable. Implements optimal linear combining under i.i.d. noise.
             Provides maximal diversity gain when separated branches are uncorrelated.

Simulation methodology:
  - Symbol-level Monte-Carlo over 2e5 realizations per SNR point.
  - Diversity order d extracted from asymptotic BER slope via log-log linear regression (SNR >= 10 dB).
  - Follows standard IEEE/3GPP practices for initial architecture verification of diversity receivers.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 90)
print("PUBLICATION VERIFICATION: PSID ARCHITECTURE (v6.1)")
print("=" * 90)
print(f"Python: {__import__('sys').version.split()[0]} | NumPy: {np.__version__}")
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("-" * 90)

SEED = 42
np.random.seed(SEED)

N_SYMBOLS = 200000
SNR_DB_RANGE = np.arange(6, 20, 2)

# ADC impairment parameters (typical for high-speed interleaved converters)
GAIN_MISMATCH = 0.020
TIMING_SKEW = 0.015

# Construct true mixing matrix and calibrated inverse
alpha = GAIN_MISMATCH / 2
beta = TIMING_SKEW * 0.5j
H_TRUE = np.array([[1, 1], 
                   [1 + alpha + beta, -1 + alpha - beta]], dtype=complex)
H_CAL  = np.array([[1, 1], 
                   [1 + 0.003, -0.997]], dtype=complex)
H_INV  = np.linalg.inv(H_CAL)

print("SIMULATION PARAMETERS:")
print(f"  Symbols per SNR point : {N_SYMBOLS:,}")
print(f"  SNR sweep             : {SNR_DB_RANGE.tolist()} dB")
print(f"  H_TRUE condition      : {np.linalg.cond(H_TRUE):.3f}")
print(f"  H_CAL condition       : {np.linalg.cond(H_CAL):.3f}")
print(f"  ADC impairments       : Gain={GAIN_MISMATCH*100:.1f}%, Skew={TIMING_SKEW*100:.1f}%")
print("-" * 90)

def gen_qpsk():
    """QPSK symbol generation with unit average power (Es = 1)"""
    b = np.random.randint(0, 2, 2)
    return (1/np.sqrt(2)) * ((2*b[0]-1) + 1j*(2*b[1]-1))

def calc_ber(tx, rx):
    """Hard-decision BER calculation for QPSK constellation"""
    tx, rx = np.asarray(tx), np.asarray(rx)
    dec = np.sign(np.real(rx)) + 1j * np.sign(np.imag(rx))
    err = np.sum(np.sign(np.real(dec)) != np.sign(np.real(tx))) + \
          np.sum(np.sign(np.imag(dec)) != np.sign(np.imag(tx)))
    return err / (2 * len(tx))

def run_baseline():
    """SISO reference: single ADC, zones aliased into effective channel h = h1 + h2"""
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
    """Theoretical upper bound: two independent RF chains, ideal MRC combining"""
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
    """Proposed architecture: dual-ADC interleaved sampling, digital separation, MRC"""
    bers = []
    for snr in SNR_DB_RANGE:
        noise_var = 10**(-snr/10)
        tx, rx = [], []
        for _ in range(N_SYMBOLS):
            s = gen_qpsk()
            h1 = (np.random.randn()+1j*np.random.randn())/np.sqrt(2)
            h2 = (np.random.randn()+1j*np.random.randn())/np.sqrt(2)
            v = np.array([h1*s, h2*s])
            
            # Analog aliasing at ADC inputs + thermal/quantization noise
            n_adc = np.sqrt(noise_var/2) * (np.random.randn(2) + 1j*np.random.randn(2))
            y = H_TRUE @ v + n_adc
            
            # Digital zone separation via calibrated inverse matrix
            v_sep = H_INV @ y
            
            # Independent pilot noise (models OFDM pilot/data multiplexing)
            n_pilot = np.sqrt(noise_var/2) * (np.random.randn(2) + 1j*np.random.randn(2))
            v_p = H_INV @ (H_TRUE @ np.array([h1, h2])) + n_pilot
            h1_est, h2_est = v_p[0], v_p[1]
            
            # MRC combining weights and decision variable
            den = np.abs(h1_est)**2 + np.abs(h2_est)**2 + 1e-12
            s_out = (np.conj(h1_est)*v_sep[0] + np.conj(h2_est)*v_sep[1]) / den
            
            tx.append(s)
            rx.append(s_out)
        bers.append(calc_ber(np.array(tx), np.array(rx)))
    return np.array(bers)

def est_diversity_order(snr, ber, n_tx, label):
    """Extract diversity order d from asymptotic BER slope (log-log regression)"""
    ber_safe = np.maximum(ber, 0.5 / n_tx)
    mask = (snr >= 10) & (ber_safe <= 0.08)
    snr_m, ber_m = snr[mask], ber_safe[mask]
    
    print(f"\n[{label}] Asymptotic region for diversity estimation:")
    print(f"   SNR [dB] : {' '.join(f'{x:3.0f}' for x in snr_m)}")
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
    
    print(f"   Result   : d = {d:.3f} +/- {se:.3f} | R^2 = {r2:.3f} | p = {p_val:.2e}")
    return d, se, r2, p_val

def main():
    print("Starting Monte-Carlo simulation (instantaneous LS estimation, independent noise)...")
    t0 = time.time()
    b_base = run_baseline()
    b_ideal = run_ideal_mrc()
    b_psid = run_psid()
    print(f"Completed in {time.time()-t0:.1f} seconds\n")
    
    d_b, se_b, r2_b, p_b = est_diversity_order(SNR_DB_RANGE, b_base, N_SYMBOLS, "Baseline")
    d_i, se_i, r2_i, p_i = est_diversity_order(SNR_DB_RANGE, b_ideal, N_SYMBOLS, "Ideal MRC")
    d_p, se_p, r2_p, p_p = est_diversity_order(SNR_DB_RANGE, b_psid, N_SYMBOLS, "PSID")
    
    plt.figure(figsize=(8,5))
    plt.semilogy(SNR_DB_RANGE, b_base, 'b--o', label='Baseline (SISO)', linewidth=2)
    plt.semilogy(SNR_DB_RANGE, b_psid,  'r-s', label='PSID (2 ADC, H_CAL, MRC)', linewidth=2)
    plt.semilogy(SNR_DB_RANGE, b_ideal, 'g-.^', label='Ideal MRC (reference)', linewidth=2)
    plt.grid(True, alpha=0.3, which='both')
    plt.xlabel('SNR, dB'); plt.ylabel('BER'); plt.yscale('log'); plt.ylim(1e-5, 0.5)
    plt.title('PSID Architecture: Frequency Diversity Order Verification')
    plt.legend(); plt.tight_layout()
    plt.savefig('psid_final_v6.1.png', dpi=300)
    print("\nFigure saved: psid_final_v6.1.png")
    
    print("\nVALIDATION CRITERIA:")
    checks = [
        (1.75 <= d_p <= 2.15, "d_proposed in [1.75, 2.15] (impairments accounted)"),
        (p_p < 0.005, "p-value < 0.005"),
        (r2_p > 0.97, "R^2 > 0.97"),
        (d_p > d_b + 0.7, "Delta d > 0.7 (improvement over SISO)"),
        (se_p < 0.10, "std_err < 0.10")
    ]
    ok = all(c[0] for c in checks)
    for c, txt in checks: 
        status = "[PASS]" if c else "[FAIL]"
        print(f"  {status} {txt}")
    
    print("=" * 90)
    result_msg = "READY FOR PEER REVIEW" if ok else "REQUIRES ADJUSTMENT"
    print(f"CONCLUSION: {result_msg}")
    print("=" * 90)

if __name__ == "__main__":
    main()
