def get_precoeff(N, de, Gamma, omega, gamma):

    Gamma_tot = Gamma + gamma * de**2/omega

    return -1j * N * omega * Gamma_tot / (Gamma_tot + 1j* N * omega)