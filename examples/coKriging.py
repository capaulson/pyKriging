"""
One-variable co-kriging example of Forrester, Sobester & Keane (2007), "Multi-fidelity optimization via surrogate
modelling", Proc. R. Soc. A 463:3251, section 3a (figures 1 and 2): four expensive samples plus eleven cheap ones
give a far better model than kriging the expensive data alone.
"""
import numpy as np
import matplotlib.pyplot as plt

from pyKriging import kriging
from pyKriging.coKriging import coKriging


def expensive(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def cheap(x, A=0.5, B=10, C=-5):
    return A * expensive(x) + B * (x - 0.5) - C


def main():
    Xc = np.linspace(0, 1, 11)       # cheap samples
    Xe = np.array([0, 0.4, 0.6, 1])  # expensive samples (a subset of Xc)

    ck = coKriging(Xc, cheap(Xc), Xe, expensive(Xe), seed=0)
    ck.train()
    k = kriging(Xe, expensive(Xe), seed=0)
    k.train()
    print(f'rho = {ck.rho:.3f} (scaling between the cheap and expensive data; the true value is 1/A = 2)')

    x = np.linspace(0, 1, 201)
    for name, f in (('kriging through y_e', k.predict(x)), ('co-kriging', ck.predict(x))):
        print(f'{name}: RMSE against f_e {np.sqrt(np.mean((f - expensive(x)) ** 2)):.3f}')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))
    ax1.plot(x, expensive(x), 'k', label='$f_e$')
    ax1.plot(x, cheap(x), 'k--', label='$f_c$')
    ax1.plot(x, k.predict(x), 'b:', label='kriging through $y_e$')
    ax1.plot(x, ck.predict(x), 'r', label='co-kriging')
    ax1.plot(Xe, expensive(Xe), 'ko', Xc, cheap(Xc), 'kx')
    ax1.legend()
    # predict_var is in normalised units (the range of all the y data is 1): rescale to the units of y
    ax2.plot(x, (ck.predict_var(x) * np.ptp(np.r_[cheap(Xc), expensive(Xe)])) ** 2)
    ax2.set_ylabel('estimated error $s^2(x)$')
    plt.show()


if __name__ == '__main__':
    main()
