import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from butterworth import butter_manual
from chebyshev1 import cheby1_manual


# --- PARTEA 1: SPECIFICATII SI GENERARE SEMNAL ---
delta_p = 0.5       
delta_s = 34        
fp = 10000    
fs = 14000       
Fs = 50000          

print(f"  δp (delta_p) = {delta_p} (atenuarea maximă din banda de trecere)")
print(f"  δs (delta_s) = {delta_s} (atenuarea minimă de oprire)")
print(f"  fp = {fp} Hz (frecvența de trecere)")
print(f"  fs = {fs} Hz (frecvența de oprire)")
print(f"  Fs = {Fs} Hz (frecvența de eșantionare)")

Ts = 1 / Fs
print(f"  Ts = 1/Fs = {Ts:.2e} sec = {Ts*1e6:.1f} μs (perioada de eșantionare)")

N_samples = 10000
t = np.arange(N_samples) * Ts 

print(f"  N_samples = {N_samples}")

F_util = 5000  
semnal_util = np.sin(2 * np.pi * F_util * t)

print(f"\n  Semnal util:")
print(f"    Frecvență: F_util = {F_util} Hz (< fp = {fp} Hz)")
print(f"    Amplitudine: 1.0")

F_zgomot = 20000
A_zgomot = 0.8
semnal_zgomot = A_zgomot * np.sin(2 * np.pi * F_zgomot * t)

print(f"\n  Semnal interferență:")
print(f"    Frecvență: F_zgomot = {F_zgomot} Hz (> fs = {fs} Hz)")
print(f"    Amplitudine: {A_zgomot}")
print(f"    Semnal: {A_zgomot}·sin(2π·{F_zgomot}·t)")

zgomot_alb = 0.5 * np.random.randn(N_samples)

print(f"\n  Zgomot alb:")
print(f"    Distribuție: Gaussian (μ=0, sigma=0.5)")
print(f"    Putere: ~0.25")

x_n = semnal_util + semnal_zgomot + zgomot_alb

print(f"\n  Semnal compus (x[n]):")
print(f"    x[n] = semnal_util + semnal_zgomot + zgomot_alb")
print(f"    Componente frecvențe: {F_util} Hz (util) + {F_zgomot} Hz (zgomot) + zgomot alb")


# --- PARTEA 2: CONVERSIE SPECIFICAȚII LA dB ---
Rp_dB = 20 * np.log10(delta_p)
Rs_dB = 20 * np.log10(delta_s)

Rp_abs = abs(Rp_dB)
Rs_abs = abs(Rs_dB)


# --- PARTEA 3: DESIGN FILTRE MANUAL ---
b_butter, a_butter, N_BUTTER_MAN = butter_manual(delta_p, delta_s, fp, fs, Fs)
print(f"\n  Butterworth manual:")
print(f"    N (ordin) = {N_BUTTER_MAN}")
print(f"    b length = {len(b_butter)}, a length = {len(a_butter)}")

b_cheby, a_cheby, N_CHEBY1_MAN = cheby1_manual(delta_p, delta_s, fp, fs, Fs)
print(f"\n  Chebyshev-I manual:")
print(f"    N (ordin) = {N_CHEBY1_MAN}")
print(f"    b length = {len(b_cheby)}, a length = {len(a_cheby)}")

F_nyquist = Fs / 2
Wp_scipy = fp / F_nyquist  

# Filter selectivity 
F_ss_butter = N_BUTTER_MAN / (2 * np.sqrt(2) * np.sqrt(Wp_scipy * np.pi))
F_ss_cheby = N_CHEBY1_MAN / (2 * np.sqrt(2) * np.sqrt(Wp_scipy * np.pi))

print(f"\n Selectivitate filtre:")
print(f"  Butterworth: F_ss = {F_ss_butter:.4f}")
print(f"  Chebyshev-I: F_ss = {F_ss_cheby:.4f}")

# Region of convergence (ROC)
epsilon = np.sqrt(10**(Rp_abs/10) - 1)
epsilon_paper = 2.5651

ROC_butter = epsilon_paper**(-1/N_BUTTER_MAN)
ROC_cheby = epsilon_paper**(-1/N_CHEBY1_MAN)

print(f"\n Region of Convergence (ROC):")
print(f"  Butterworth: ROC = {ROC_butter:.4f}")
print(f"  Chebyshev-I: ROC = {ROC_cheby:.4f}")

# Aplicare filtre pe semnal
y_butter = signal.filtfilt(b_butter, a_butter, x_n)
y_cheby = signal.filtfilt(b_cheby, a_cheby, x_n)

# --- PARTEA 4: VIZUALIZARE SI ANALIZA ---
def plot_filter_analysis(b, a, titlu, N, Fs, filename_prefix):
    w, h = signal.freqz(b, a, worN=8000, fs=Fs)

    plt.figure(figsize=(12, 10))
    plt.suptitle(f'Analiză filtru {titlu} (N={N})', fontsize=16)
    
    # 1. Magnitudinea liniara
    plt.subplot(2, 2, 1)
    plt.plot(w, np.abs(h), 'r')
    plt.title('Răspuns în frecvență - Magnitudine (liniar)')
    plt.xlabel('Frecvența [Hz]')
    plt.ylabel('Magnitudine')
    plt.grid()

    # 2. Magnitudinea in dB
    plt.subplot(2, 2, 2)
    plt.plot(w, 20 * np.log10(np.maximum(np.abs(h), 1e-10)), 'b')
    plt.title('Răspuns în frecvență - Magnitudine (dB)')
    plt.xlabel('Frecvența [Hz]')
    plt.ylabel('Magnitudine [dB]')
    plt.ylim(-200, 5)
    plt.grid()

    # 3. Faza
    plt.subplot(2, 2, 3)
    plt.plot(w, np.angle(h), 'g')
    plt.title('Răspuns în frecvență - Fază, rad/s')
    plt.xlabel('Frecvența [Hz]')
    plt.ylabel('Fază [radiani]')
    plt.grid()

    # 4. Planul poli-zerouri
    plt.subplot(2, 2, 4)
    z, p, k = signal.tf2zpk(b, a)

    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5)
    plt.plot(np.real(z), np.imag(z), 'go', mfc='none', label='Zerouri')
    plt.plot(np.real(p), np.imag(p), 'rx', label='Poli')
    plt.title('Planul poli-zerouri')
    plt.xlabel('Partea reală')
    plt.ylabel('Partea imaginară')
    plt.axis('equal')
    plt.legend()
    plt.grid()  

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'./sample_signal/filter_analysis_{titlu.lower().replace(" ", "_")}.png')


# Generare analize individuale
plot_filter_analysis(b_butter, a_butter, 'Butterworth', N_BUTTER_MAN, Fs, 'butterworth')
plot_filter_analysis(b_cheby, a_cheby, 'Chebyshev-I', N_CHEBY1_MAN, Fs, 'chebyshev1')

# Comparatia transition band
w_b, h_b = signal.freqz(b_butter, a_butter, worN=8000)
w_c, h_c = signal.freqz(b_cheby, a_cheby, worN=8000)

plt.figure(figsize=(12, 5))

# Magnitudine liniara normalizata
plt.subplot(1, 2, 1)
plt.plot(w_b/np.pi, np.abs(h_b), 'g', label=f'Butterworth (N={N_BUTTER_MAN})')
plt.plot(w_c/np.pi, np.abs(h_c), 'r', label=f'Chebyshev-I (N={N_CHEBY1_MAN})')
plt.title('Comparatie Răspuns în frecvență - Magnitudine (liniar)')
plt.xlabel('Frecvența normalizată')
plt.ylabel('Magnitudine')
plt.legend()
plt.grid()

# Magnitudine in dB normalizata
plt.subplot(1, 2, 2)
plt.plot(w_b/np.pi, 20 * np.log10(np.maximum(np.abs(h_b), 1e-15)), 'g', label=f'Butterworth (N={N_BUTTER_MAN})')
plt.plot(w_c/np.pi, 20 * np.log10(np.maximum(np.abs(h_c), 1e-15)), 'r', label=f'Chebyshev-I (N={N_CHEBY1_MAN})')
plt.title('Comparatie Răspuns în frecvență - Magnitudine (dB)')
plt.xlabel('Frecvența normalizată')
plt.ylabel('Magnitudine [dB]')
plt.ylim(-400, 10)
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('./sample_signal/filter_comparison_transition_band.png', dpi=300, bbox_inches='tight')

# --- PARTEA 5: VIZUALIZARE REZULTAT FILTRARE (DOMENIUL TIMP ȘI FRECVENȚĂ) ---
def plot_signal_comparison(t, x, y, titlu, filename):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(t[:500] * 1000, x[:500], 'b', label='Semnal corupt (Input)')
    plt.plot(t[:500] * 1000, y[:500], 'r', label='Semnal filtrat (Output)', linewidth=2)
    plt.title(f'Rezultat filtrare {titlu} - Domeniul timp')
    plt.xlabel('Timp [ms]')
    plt.ylabel('Amplitudine')
    plt.legend()

    plt.subplot(1, 2, 2)
    N = len(x)
    xf = fftfreq(N, Ts)[:N//2]

    X_fft = 2.0/N * np.abs(fft(x)[:N//2])
    Y_fft = 2.0/N * np.abs(fft(y)[:N//2])

    plt.plot(xf/1000, X_fft, alpha=0.5, label='Input spectrum')
    plt.plot(xf/1000, Y_fft, label='Output spectrum')
    plt.axvline(fp/1000, color='r', linestyle='--', label='passband limit')
    plt.title(f'Analiza spectrală după filtrare {titlu} - Domeniul frecvență')
    plt.xlabel('Frecvența [kHz]')
    plt.ylabel('Magnitudine')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')

plot_signal_comparison(t, x_n, y_butter, 'Butterworth', './sample_signal/signal_comparison_butterworth.png')
plot_signal_comparison(t, x_n, y_cheby, 'Chebyshev-I', './sample_signal/signal_comparison_chebyshev1.png')
plt.show()
