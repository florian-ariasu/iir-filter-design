import numpy as np
from scipy import signal

def cheby1_manual(delta_p, delta_s, fp, fs, Fs):
    # 1. Conversie specificații în dB
    Rp = 20 * np.log10(delta_p)
    Rs = 20 * np.log10(delta_s)
    Rp_abs = abs(Rp)
    Rs_abs = abs(Rs)

    # 2. Parametru epsilon pentru ondulații (Ecuația 27)
    eps = np.sqrt(10**(Rp_abs/10) - 1)   

    # 3. Calcul ordin filtru N (Ecuația 9)
    num = np.arccosh(np.sqrt((10**(Rs_abs/10) - 1) / (10**(Rp_abs/10) - 1)))
    den = np.arccosh(fs/fp)
    N = int(np.ceil(num / den))          

    # 4. Prewarping pentru frecvența de tăiere
    T = 1.0 / Fs
    Omega_p = 2 * np.pi * fp / Fs
    wp_tilde = (2 / T) * np.tan(Omega_p / 2)

    # 5. Calcul poli pe elipsă (Ecuația 25)
    alpha = np.arcsinh(1/eps) / N
    poles = []
    for k in range(1, N+1):
        theta = np.pi * (2*k - 1) / (2*N)
        sigma = -np.sinh(alpha) * np.sin(theta)
        omega =  np.cosh(alpha) * np.cos(theta)
        p = wp_tilde * (sigma + 1j*omega)
        if np.real(p) < 0: # Verificare stabilitate (în interiorul ROC)
            poles.append(p)
    p_analog = np.array(poles)

    # 6. Calcul câștig pentru a compensa ripple-ul
    z_analog = np.array([])
    s0 = 1j * 0.0
    H0 = 1.0 / np.prod(s0 - p_analog)
    target = 1.0 / np.sqrt(1 + eps**2) if N % 2 == 0 else 1.0
    K = target / np.abs(H0)

    # 7. Aplicare Transformare Biliniară (S -> Z)
    zd, pd, kd = signal.bilinear_zpk(z_analog, p_analog, K, fs=Fs)
    
    # 8. Conversie în funcție de transfer (coeficienți b și a)
    b_digital, a_digital = signal.zpk2tf(zd, pd, kd)

    # 9. Normalizare
    a0 = a_digital[0]
    b_digital = (b_digital / a0).real
    a_digital = (a_digital / a0).real

    # 10. Ajustare finală câștig (DC Gain)
    w0, h0 = signal.freqz(b_digital, a_digital, worN=[0], fs=Fs)
    gain0 = np.abs(h0[0])
    b_digital = b_digital / gain0

    return b_digital, a_digital, N
