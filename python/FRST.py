import numpy as np
import scipy
from scipy import special


def nufft_check_dft(gamma, Nd):
    M = np.size(gamma)
    Nd = 64

    kk = gamma/(2*np.pi) * Nd
    # kk = gamma/(2*np.pi) * np.tile(Nd[:], (M, 1)] simile al testo ma qui non ha senso dato che Nd e' un intero

    tol = 1e-6
    tmp = np.abs(np.around(kk) - kk)
    if np.all(tmp[:] < tol) and np.any(gamma):
        print('DFT samples has suboptimal accuracy')

def kaiser_bessel(u,J):
    #if J is None:
    #    J = 6
    d = 1
    kb_m = 0
    alpha = 2.34 * J
    z = np.sqrt(np.power(2*np.pi * (J/2)*u,2)-np.power(alpha,2)+0j)
    nu = d/2 + kb_m
    y = np.power(2*np.pi, d/2) * np.power(J/2, d) * np.power(alpha, kb_m) / scipy.special.iv(kb_m, alpha) * \
        scipy.special.jv(nu, z) / np.power(z, nu)
    return np.real(y)

def nufft_alpha_kb_fit(N, J, K):
    beta = 1
    argNmid = (N-1)/2

    if N > 40:
        argL = 13 # empirically found to be reasonable
    else:
        argL = np.ceil(N/3) # a kludge to avoid "rank deficient" complaints

    nlist = np.arange(0,N) - argNmid

    # kaiser-bessel with previously numerically-optimized shape
    sn_kaiser = 1/kaiser_bessel(nlist/K, J)

    # use regression to match NUFFT with BEST kaiser scaling's
    gam = 2 * np.pi /K
    X = np.cos(beta*gam*np.outer(nlist,np.arange(0, argL+1)))
    coef,residuals, rank, s = np.linalg.lstsq(X, sn_kaiser, rcond=None)

    if np.any(np.isnan(coef)): # if any NaN then big problem!
        coef = np.linalg.pinv(X)@sn_kaiser
        if np.any(np.isnan(coef)):
            raise ValueError('bug: NaN coefficients')

    alphas = np.append(np.real(coef[0]), np.real(coef[1:])/2)
    return alphas, beta

def nufft_scale(Nd, Kd, alpha, beta ,Nmid):
    if not np.isreal(alpha[0]):
        raise ValueError('alpha[0] needs to be real')
    L = np.size(alpha)-1 #check

    if L > 0:
        sn = np.zeros((Nd, 1))
        n = np.arange(0, Nd)
        i_gam_n_n0 = 1j * (2*np.pi/Kd) * (n-Nmid) * beta
        i_gam_n_n0 = i_gam_n_n0.reshape((Nd, 1))  # Reshape 1D to 2D array (specific to numpy)

        for l1 in range(-L, L+1):
            alf = alpha[abs(l1)]
            if l1 < 0:
                alf = np.conj(alf)
            sn = sn + (alf * np.exp(i_gam_n_n0 * l1))
    else:
        sn = alpha * np.ones((Nd, 1))
    return sn


def nufft_T(N, J, K, tol, alpha, beta):

    if N > K:
        raise ValueError('N > K error')
    try:
        alpha
    except NameError:
        j2, j1 = np.meshgrid(np.arange(1, J + 1), np.arange(1, J + 1))
        cssc = np.sinc((j2 - j1) / (K / N))

    if alpha is None:
        j2, j1 = np.meshgrid(np.arange(1, J + 1), np.arange(1, J + 1))
        cssc = np.sinc((j2 - j1) / (K / N))

    # Fourier-series based scaling factors
    else:
        if not np.isreal(alpha[0]):
            raise ValueError('alpha[0] must be real')
        L = np.size(alpha) - 1
        cssc = np.zeros((J, J))
        j2, j1 = np.meshgrid(np.arange(1, J + 1), np.arange(1, J + 1))

        for l1 in range(-L, L + 1):
            for l2 in range(-L, L + 1):
                alf1 = alpha[abs(l1)]
                if l1 < 0:
                    alf1 = np.conj(alf1)
                alf2 = alpha[abs(l2)]
                if l2 < 0:
                    alf2 = np.conj(alf2)
                tmp = j2 - j1 + beta * (l1 - l2)
                tmp = np.sinc(tmp / (K / N))
                cssc = cssc + alf1 * np.conj(alf2) * tmp

    # Inverse (or pseudo inverse)

    u, s, vh = np.linalg.svd(cssc)
    smin = np.min(s)

    if smin < tol:
        print('Warning, poor conditioning ' + str(smin) + ' -> pinverse')
        T = np.linalg.pinv(cssc, rcond=tol / 10)
    else:
        T = np.linalg.pinv(cssc)

    return T

def nufft_r(om, N, J, K, alpha, beta):

    try:
        alpha
    except NameError and alpha is None:
        alpha = 1  # default Fourier series coefficients of scaling factors

    if alpha is None:
        alpha = 1  # default Fourier series coefficients of scaling factors

    try:
        beta
    except NameError and alpha is None:
        beta = 0.5  # default is Liu version for now

    if beta is None:
        beta = 0.5  # default is Liu version for now

    M = np.size(om)
    gam = 2 * np.pi / K
    dk = om / gam - np.floor(om / gam - J / 2)
    arg = np.add.outer(-np.arange(1, J + 1), dk)

    L = np.size(alpha) - 1  # check

    if L > 0:
        rr = np.zeros((J, M))

        for l1 in range(-L, L + 1):
            alf = alpha[abs(l1)]
            if l1 < 0:
                alf = np.conj(alf)
            r1 = np.sinc((arg + l1 * beta) / (K / N))
            rr = rr + alf * r1
    else:
        rr = np.sinc(arg / (K / N))

    return rr, arg


def nufft_interpMatrixKBminMax(gamma, Nd, Jd, Kd):

    M = np.size(gamma)

    nufft_check_dft(gamma, Nd)
    alpha = None
    alpha, beta = nufft_alpha_kb_fit(Nd, Jd, Kd)

    # Find NUFFT scaling factor

    Nmid = (Nd - 1) / 2
    tmp = nufft_scale(Nd, Kd, alpha, beta, Nmid)
    sn = scipy.sparse.coo_matrix((tmp.reshape(Nd), (np.arange(0, Nd), np.arange(0, Nd))), shape=(Nd, Nd))

    # NUFFT interpolation matrix
    tol = 0
    T = nufft_T(Nd, Jd, Kd, tol, alpha, beta)
    r, arg = nufft_r(gamma[:], Nd, Jd, Kd, alpha, beta);
    c = T @ r
    # T = None
    # r = None
    gam = 2 * np.pi / Kd
    phase_scale = 1j * gam * (Nd - 1) / 2

    phase = np.exp(phase_scale * arg)
    uu = np.conj(phase * c)

    # indices into oversampled FFT components
    koff = np.floor(gamma[:] / gam - Jd / 2)
    kk = np.mod(np.add.outer(np.arange(1, Jd + 1), koff), Kd)

    mm = np.arange(0, M)
    mm = np.tile(mm, (np.prod(Jd), 1))

    sp = scipy.sparse.coo_matrix((uu.ravel(), (mm.ravel(), kk.ravel())), shape=(M, np.prod(Kd)))

    return Nd, Kd, M, sn, sp



def nufft_interpMatrixNN(gamma,Nd,Kd):
    M  = np.size(gamma)

    # NUFFT scaling factor
    sn = scipy.sparse.identity(Nd)

    # NUFFT interpolation matrix

    positiveInput = gamma > 0
    gamma = np.mod(gamma, 2*np.pi)
    gamma[(gamma == 0) & positiveInput] = 2*np.pi
    DFT_w = np.linspace(0, 2*np.pi/Kd * (Kd-1), Kd)
    tmp = np.add.outer(gamma, -DFT_w)
    idx_min = np.argmin(np.abs(tmp), axis=1)

    sp = scipy.sparse.coo_matrix((np.ones(M), (np.arange(0, M), idx_min)), shape=(M, Kd))
    return Nd, Kd, M, sn, sp


def nufft(x, Kd,sp):
    # NUFFT
    Xk = np.fft.fft(x, Kd, axis=0)
    X = sp @ Xk

    return X


# FRST
def frst(p, f, c, d, L, mbar, W, qbar, sigma, N, B):

    # Parameters
    z = np.linspace(0, d*(L-1), num=L).T                                  # [L,1] microphone positions
    m = np.linspace(-(W-1)/2*mbar, ((W-1)*mbar-((W-1)/2*mbar)), num=W).T  # [W,1] m axis
    q = np.linspace(0, z[-1], num=np.around(z[-1]/qbar) + 1).T            # [j,1] q axis
    I = np.size(q)                                                        # Number of frames

    # Signal windowing
    arg_psi = np.add.outer(z,-q)
    psi = np.exp(-np.pi*np.power(arg_psi, 2) / np.power(sigma, 2))        # Gaussian window with std sigma

    # Set of uniformly spaced frequency locations
    gamma = (2*np.pi*f/c)*(m/np.sqrt(1+np.power(m, 2)))*d

    if B == 1:
        Nd, Kd, M, sn, sp = nufft_interpMatrixNN(gamma, L, N)
    else:
        Nd, Kd, M, sn, sp = nufft_interpMatrixKBminMax(gamma, L, B, N)

    pbar = np.tile(d*sn@p, (1, I)) * psi


    Z = nufft(pbar, Kd, sp).T

    return Z, m, q, z



