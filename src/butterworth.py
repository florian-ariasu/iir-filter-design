import numpy as np
from scipy import signal

def butter_manual(delta_p, delta_s, fp, fs, Fs):
    # 1. Conversie specificații în dB (Ecuațiile 12, 13)
    Rp = 20 * np.log10(delta_p)   
    Rs = 20 * np.log10(delta_s)
    Rp_abs = abs(Rp)
    Rs_abs = abs(Rs)

    # 2. Frecvențe unghiulare analogice
    wp = 2 * np.pi * fp
    ws = 2 * np.pi * fs

    # 3. Calcul ordin filtru N (Ecuația 8)
    N = np.log10((10**(Rs_abs/10) - 1) / (10**(Rp_abs/10) - 1)) / np.log10(ws/wp)
    N = int(np.ceil(N))

    # 4. Prewarping frecvență pentru Transformarea Biliniară (Ecuația 19)
    T = 1.0 / Fs
    Omega_p = 2 * np.pi * fp / Fs
    wp_tilde = (2 / T) * np.tan(Omega_p / 2)

    # 5. Calcul poziție poli în planul S (Ecuația 21)
    poles = []
    for k in range(N):
        theta = np.pi * (2*k + 1 + N) / (2*N)
        p = np.exp(1j * theta)     
        if np.real(p) < 0: # Păstrăm doar polii din semiplanul stâng (stabilitate)
            poles.append(p)
    poles = np.array(poles) * wp_tilde

    # 6. Definire model analogic (fără zerouri)
    z_analog = np.array([])
    p_analog = poles
    K = np.prod(-p_analog).real

    # 7. Transformarea Biliniară din planul S în planul Z (Ecuația 4)
    zd, pd, kd = signal.bilinear_zpk(z_analog, p_analog, K, fs=Fs)

    # 8. Obținere coeficienți b (numărător) și a (numitor) (Ecuația 3)
    b_digital, a_digital = signal.zpk2tf(zd, pd, kd)

    # 9. Normalizare coeficienți
    a0 = a_digital[0]
    b_digital = (b_digital / a0).real
    a_digital = (a_digital / a0).real

    # 10. Corecție câștig în curent continuu (DC gain = 1)
    w0, h0 = signal.freqz(b_digital, a_digital, worN=[0], fs=Fs)
    b_digital = b_digital / np.abs(h0[0])

    return b_digital, a_digital, N
