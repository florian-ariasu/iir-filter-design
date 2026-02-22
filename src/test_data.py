import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

from butterworth import butter_manual
from chebyshev1 import cheby1_manual


data = pd.read_csv('100_ekg.csv')   

ecg = data['MLII'].values          
Fs_real = 360             
Ts_real = 1.0 / Fs_real
t_real  = np.arange(len(ecg)) * Ts_real

delta_p_real = 0.5
delta_s_real = 34
fp_real = 40.0
fs_real = 60.0   

bB_ecg, aB_ecg, N_B_ecg = butter_manual(delta_p_real, delta_s_real, fp_real, fs_real, Fs_real)
bC_ecg, aC_ecg, N_C_ecg = cheby1_manual(delta_p_real, delta_s_real, fp_real, fs_real, Fs_real)

yB_ecg = signal.filtfilt(bB_ecg, aB_ecg, ecg)
yC_ecg = signal.filtfilt(bC_ecg, aC_ecg, ecg)

def plot_real_comparison(t, x, y, fp, Fs, titlu):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    durata = 5
    n_show = min(int(durata * Fs), len(x))
    plt.plot(t[:n_show], x[:n_show], color='C0', linewidth=0.8, label='ECG original')
    plt.plot(t[:n_show], y[:n_show], color='C1', linewidth=1.5, label='ECG filtrat')
    plt.xlabel('Timp [s]')
    plt.ylabel('Amplitudine [mV]')
    plt.title(f'Rezultat filtrare {titlu} (primele {durata}s)')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)

    N = len(x)
    xf = fftfreq(N, 1.0/Fs)[:N//2]
    X_fft = 2.0/N * np.abs(fft(x)[:N//2])
    Y_fft = 2.0/N * np.abs(fft(y)[:N//2])

    plt.subplot(1, 2, 2)
    plt.semilogy(xf, np.maximum(X_fft, 1e-8), color='C0', alpha=0.5, label='Original')
    plt.semilogy(xf, np.maximum(Y_fft, 1e-8), color='C1', label='Filtrat')
    plt.axvline(fp, color='r', linestyle='--', linewidth=1.2, label='fp')
    plt.xlim(0, 150)
    plt.xlabel('Frecvența [Hz]')
    plt.ylabel('Magnitudine (log)')
    plt.title(f'Spectru {titlu}')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(f'./ecg_signal/ecg_{titlu.lower()}.png', dpi=300, bbox_inches='tight')

plot_real_comparison(t_real, ecg, yB_ecg, fp_real, Fs_real, 'Butterworth')
plot_real_comparison(t_real, ecg, yC_ecg, fp_real, Fs_real, 'Chebyshev-I')
plt.show()
