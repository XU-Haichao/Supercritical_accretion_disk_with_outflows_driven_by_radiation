# This is a target code that is called by other code to find inner boundary conditions.
# The shooting method is implemented by fsolve function in scipy module.
import numpy as np
from numba import jit
from math import *
from scipy.optimize import fsolve
from scipy.integrate import cumtrapz, simpson

kappa = 0.4
gamma = polyindex = 1.5
r_in = 6
h_max = 2 ** 0.5 / 2
lw_out = 0
xi_out = 1.5
omega_in = 1.

num = 2000


def f_rad_crit(r):  # Critical radiation of any radius.
    return (2 * 3 ** 0.5) / (9 * r ** 2)


def h(f_rad_H, r):  # Half thickness of accretion disk
    def func_h(h):
        return f_rad_H - h / (r ** 2 * (1 + h ** 2) ** 1.5)

    if f_rad_H < f_rad_crit(r):
        return float(fsolve(func_h, 0))
    else:
        return h_max


def outflows(r, tau_H, f_rad_H):
    tau_z = np.linspace(0, tau_H, num).reshape(num)

    def dGamma_dtau(tau_z):
        return np.power((1 + 3 / 8 * tau_H - (3 * tau_z ** 2) / (8 * tau_H)), -1 / gamma)

    Gamma_ = dGamma_dtau(tau_z)
    Gamma = cumtrapz(Gamma_, tau_z, initial=0)
    Gamma_H = Gamma[-1]

    rho_H = Gamma_H / tau_H
    z = Gamma / Gamma_H * h_max
    f_rad_z = ((tau_z / tau_H) * f_rad_H)
    vz_H = abs(simpson(2 * (r ** 2 * f_rad_z - z / (1 + z ** 2) ** 1.5), z)) ** 0.5
    return np.array([float(rho_H), float(vz_H)])


@jit
def B(h):
    return 1 / (1 + h) ** 1.5


@jit
def omega(frad, r, h, B):
    return sqrt(1 - 0.5 * h * B * r ** 2 * frad)


def target(alpha, dotm_out, r_out, dotm_in=1.1, lw_in=10.):
    def target_test(x):
        dotm_in, lw_in = x

        def f_rad_out(dotm_out):
            def fun_frad_out(frad_out):
                frad_crit_out = f_rad_crit(r_out)
                h_out = h(frad_out, r_out)
                B_out = (1 + h_out ** 2) ** (-1.5)
                omega_out = (1 - 0.5 * h_out * B_out * r_out ** 2 * frad_out) ** 0.5
                f_out = 1 - (1 - lw_in) * (
                        (dotm_in * r_in ** 0.5 * omega_in) / (dotm_out * r_out ** 0.5 * omega_out))  # 无量纲化的公式(33)
                Qadv_out = (15 * dotm_out * omega_out ** 2 / r_out ** 3) * f_out  # 公式(54)

                if frad_out < frad_crit_out:
                    Qw_out = 0
                else:
                    tau_out = (20 * sqrt(2) / 3) * (dotm_out * f_out) / (
                            alpha * h_out ** 1.5 * r_out ** 1.5 * B_out ** 0.5 * frad_out ** 0.5)  # 公式(49)
                    outflows_out = outflows(r_out, tau_out, frad_out)
                    rho_out = outflows_out[0]
                    vz_out = outflows_out[1]
                    Qw_out = (rho_out * vz_out ** 3 * tau_out) / (2 * h_out * r_out ** 2.5)  # 公式(58)
                test = (Qadv_out - Qw_out) / ((5 * dotm_out * h_out * xi_out * B_out) / r_out + 1)
                return test - frad_out

            return float(fsolve(fun_frad_out, 1e-10))

        def derivative(lnfrad_lndotm_lw, lnr_d):
            lnfrad_d = lnfrad_lndotm_lw[0]
            lndotm_d = lnfrad_lndotm_lw[1]
            lw_d = lnfrad_lndotm_lw[2]
            r_d = exp(lnr_d)
            frad_d = exp(lnfrad_d)
            dotm_d = exp(lndotm_d)
            h_d = h(frad_d, r_d)
            B_d = B(h_d)
            omega_d = omega(frad_d, r_d, h_d, B_d)
            f_d = f(lw_d, dotm_d, r_d, omega_d)
            if frad_d < f_rad_crit(r_d):
                dlndotm_dlnr = 0
                dlw_dlnr = 0
                dlnf_dlnr = 0.5 * (1 - lw_in + lw_d) * (
                        (dotm_in * sqrt(r_in) * omega_in) / (f_d * dotm_d * sqrt(r_d) * omega_d))
                dlnfrad_dlnr = -11 / (6 - h_d ** 2) + (2 * (1 - 2 * h_d ** 2) * dlnf_dlnr) / (3 * (6 - h_d ** 2)) - (
                        2 * (1 - 2 * h_d ** 2) * omega_d ** 2 * f_d) / (
                                       (6 - h_d ** 2) * h_d * r_d ** 2 * B_d * frad_d + 1e-15) + (
                                       2 * (1 - 2 * h_d ** 2) * r_d) / (
                                           15 * (6 - h_d ** 2) * h_d * dotm_d * B_d + 1e-15)
                return [dlnfrad_dlnr, dlndotm_dlnr, dlw_dlnr]
            else:
                tau_d = (20 * 2 ** 0.5 * dotm_d * f_d) / (
                        3 * alpha * h_d ** 1.5 * r_d ** 1.5 * B_d ** 0.5 * frad_d ** 0.5)
                outflows_d = outflows(r_d, tau_d, frad_d)
                rho_d = outflows_d[0]
                vz_d = outflows_d[1]
                dlw_dlnr = -(0.1 * rho_d * vz_d * omega_d * tau_d * r_d) / (dotm_d * r_in ** 0.5 * omega_in * h_d)
                dlndotm_dlnr = (2 * 2 ** 0.5 * f_d * rho_d * vz_d) / (
                        3 * alpha * h_d ** 2.5 * r_d * B_d ** 0.5 * frad_d ** 0.5)
                dlnf_dlnr = (0.5 + dlndotm_dlnr) * (1 - lw_in + lw_d) * (
                        (dotm_in * sqrt(r_in) * omega_in) / (f_d * dotm_d * sqrt(r_d) * omega_d)) - dlndotm_dlnr / f_d
                dlnfrad_dlnr = -11 / 7 + 2 * dlnf_dlnr / 7 - (6 * omega_d ** 2 * f_d) / (
                        7 * h_d * r_d ** 2 * B_d * frad_d) + (
                                       2 * r_d) / (35 * h_d * dotm_d * B_d) + 2 * dlndotm_dlnr / 7 + (
                                       2 * vz_d ** 2 * dlndotm_dlnr) / (
                                       7 * h_d * B_d * r_d ** 2 * frad_d)
                return [dlnfrad_dlnr, dlndotm_dlnr, dlw_dlnr]

        @jit
        def f(lw, dotm, r, omega):
            return 1 - (1 - lw_in + lw) * ((dotm_in * r_in ** 0.5 * omega_in) / (dotm * r ** 0.5 * omega))

        lnr, deltalnr = np.linspace(log(r_in + 0.01), log(r_out), num, retstep=True)
        y = np.zeros((num, 3))
        dy_dlnr = np.zeros((num, 3))
        y[-1] = [np.log(f_rad_out(dotm_out)), np.log(dotm_out), lw_out]

        for ii1 in range(num - 1, 0, -1):
            dy_dlnr[ii1] = derivative(y[ii1], lnr[ii1])
            y[ii1 - 1] = y[ii1] - dy_dlnr[ii1] * deltalnr

        y = np.nan_to_num(y, nan=0)
        dotm_r = np.exp(y[:, 1:2])
        lw_r = y[:, 2:3]
        dotm_in_test = np.min(dotm_r[dotm_r != 1.])
        lw_in_test = np.max(lw_r)
        print(dotm_in_test, lw_in_test)
        print(dotm_in, lw_in)
        return [dotm_in_test - dotm_in, lw_in_test - lw_in]

    y0 = np.array([dotm_in,lw_in])
    return fsolve(target_test, y0, xtol=1e-5, maxfev=int(1e5))
