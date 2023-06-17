# The model is mainly derived from (Cao & Gu 2022)
# In the scripts, we always use lowercase names for dimensionless quantities and uppercase names for real quantities

import matplotlib.pyplot as plt
from inner_target import *
from scipy.interpolate import interp1d
import matplotlib.colors as colors

# The physics quantities are in the cm-g-s units
c_const = Lightspeed = 2.998e10
sigma_const = Stefan_Boltzmann_constant = 5.67e-5
h_const = Planck_constant = 6.625e-27
k_const = Boltzmann_constant = 1.3806e-16
m_const = Solar_mass = 1.989e33
G_const = Gravitational_constant = 6.673e-8
e_const = electron_charge = 4.803e-10
V_const = Gaussian_volt = 3.3356e-3

# We use the suggested values in (Cao & Gu 2022) for these physical parameters.
kappa = 0.4
gamma = polyindex = 1.5
r_in = 6.01
lw_out = 0
xi_out = 1.5
omega_in = 1.

num = 2000  #Tthe number of samples
h_max = 2 ** 0.5 / 2  # Maximum dimensionless half thickness of accretion disk

# Please set the model parameter values here.
M = m_const * 2e7
r_out = 5000
dotm_out = 100
alpha = 0.3

dotm_in, lw_in = target(alpha, dotm_out, r_out, dotm_in=0.5, lw_in=30.)  # input the initial guess of dotm_in and lw_in
lnr, deltalnr = np.linspace(log(r_in), log(r_out), num, retstep=True)
r = np.exp(lnr)

Rg = (G_const * M) / (c_const ** 2)


# Here we derive the vertical structure of any radius.
def vertical_structure(r, tau_H, f_rad_H):
    tau_z = np.linspace(0, tau_H, num)

    def dGamma_dtau(tau_z):
        return np.power((1 + 3 / 8 * tau_H - (3 * tau_z ** 2) / (8 * tau_H)), -1 / gamma)

    Gamma_ = dGamma_dtau(tau_z)
    Gamma = cumtrapz(Gamma_, tau_z, initial=0)
    Gamma_H = Gamma[-1]

    F_rad_H = (f_rad_H * G_const * M * c_const) / (kappa * Rg ** 2)
    T_H = (F_rad_H / sigma_const) ** 0.25
    T_z = (Gamma_ / Gamma_[-1]) ** (-gamma / 4) * T_H

    h_z = h(f_rad_H, r)
    z = Gamma / Gamma_H * h_z

    rho_H = Gamma_H / tau_H
    if f_rad_H <= f_rad_crit(r):
        rho_z = (1 + h_z ** 2) ** 1.5 * (1 - 2 * z ** 2) / (1 + z ** 2) ** 2.5
        f_rad_z = z / (r ** 2 * (1 + z ** 2) ** 1.5)
    else:
        rho_z = rho_H / Gamma_
        f_rad_z = ((tau_z / tau_H) * f_rad_H)
    Pres_z = (4 * sigma_const * T_z ** 4) / (3 * c_const)
    vz_z = cumtrapz(2 * (r ** 2 * f_rad_z - z / (1 + z ** 2) ** 1.5), z, initial=0) ** 0.5
    return np.array([z, f_rad_z, vz_z, rho_z, T_z, Pres_z, tau_z])


def f_rad_out(dotm_out):
    def fun_frad_out(frad_out):
        frad_crit_out = f_rad_crit(r_out)
        h_out = h(frad_out, r_out)
        B_out = (1 + h_out ** 2) ** (-1.5)
        omega_out = (1 - 0.5 * h_out * B_out * r_out ** 2 * frad_out) ** 0.5
        f_out = 1 - (1 - lw_in) * (
                (dotm_in * r_in ** 0.5 * omega_in) / (dotm_out * r_out ** 0.5 * omega_out))
        Qadv_out = (15 * dotm_out * omega_out ** 2 / r_out ** 3) * f_out

        if frad_out < frad_crit_out:
            Qw_out = 0
        else:
            tau_out = (20 * sqrt(2) / 3) * (dotm_out * f_out) / (
                    alpha * h_out ** 1.5 * r_out ** 1.5 * B_out ** 0.5 * frad_out ** 0.5)
            outflows_out = outflows(r_out, tau_out, frad_out)
            rho_out = outflows_out[0]
            vz_out = outflows_out[1]
            Qw_out = (rho_out * vz_out ** 3 * tau_out) / (2 * h_out * r_out ** 2.5)
        test = (Qadv_out - Qw_out) / ((5 * dotm_out * h_out * xi_out * B_out) / r_out + 1)
        return test - frad_out

    return float(fsolve(fun_frad_out, 1e-10))


@jit
def f(lw, dotm, r, omega):
    return 1 - (1 - lw_in + lw) * ((dotm_in * r_in ** 0.5 * omega_in) / (dotm * r ** 0.5 * omega))


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
                               2 * (1 - 2 * h_d ** 2) * r_d) / (15 * (6 - h_d ** 2) * h_d * dotm_d * B_d + 1e-15)
        return [dlnfrad_dlnr, dlndotm_dlnr, dlw_dlnr]
    else:
        tau_d = (20 * 2 ** 0.5 * dotm_d * f_d) / (3 * alpha * h_d ** 1.5 * r_d ** 1.5 * B_d ** 0.5 * frad_d ** 0.5)
        outflows_d = outflows(r_d, tau_d, frad_d)
        rho_d = outflows_d[0]
        vz_d = outflows_d[1]
        dlw_dlnr = -(0.1 * rho_d * vz_d * omega_d * tau_d * r_d) / (dotm_d * r_in ** 0.5 * omega_in * h_d)
        dlndotm_dlnr = (2 * 2 ** 0.5 * f_d * rho_d * vz_d) / (3 * alpha * h_d ** 2.5 * r_d * B_d ** 0.5 * frad_d ** 0.5)
        dlnf_dlnr = (0.5 + dlndotm_dlnr) * (1 - lw_in + lw_d) * (
                (dotm_in * sqrt(r_in) * omega_in) / (f_d * dotm_d * sqrt(r_d) * omega_d)) - dlndotm_dlnr / f_d
        dlnfrad_dlnr = -11 / 7 + 2 * dlnf_dlnr / 7 - (6 * omega_d ** 2 * f_d) / (7 * h_d * r_d ** 2 * B_d * frad_d) + (
                2 * r_d) / (35 * h_d * dotm_d * B_d) + 2 * dlndotm_dlnr / 7 + (2 * vz_d ** 2 * dlndotm_dlnr) / (
                               7 * h_d * B_d * r_d ** 2 * frad_d)
        return [dlnfrad_dlnr, dlndotm_dlnr, dlw_dlnr]


y = np.zeros((num, 3))
dy_dlnr = np.zeros((num, 3))
y[-1] = [np.log(f_rad_out(dotm_out)), np.log(dotm_out), lw_out]

# Here we use the Euler iteration to derive the radial structure of the accretion disk.
for ii1 in range(num - 1, 0, -1):
    dy_dlnr[ii1] = derivative(y[ii1], lnr[ii1])
    y[ii1 - 1] = y[ii1] - dy_dlnr[ii1] * deltalnr

y = np.nan_to_num(y, nan=0)
f_rad_r = np.exp(y[:, 0:1]).reshape(num)
dotm_r = np.exp(y[:, 1:2]).reshape(num)
lw_r = y[:, 2:3].reshape(num)

h_r = np.zeros(num)
B_r = np.zeros(num)
omega_r = np.zeros(num)
f_r = np.zeros(num)
for ii2 in range(num):
    h_r[ii2] = h(f_rad_r[ii2], r[ii2])
    B_r[ii2] = B(h_r[ii2])
    omega_r[ii2] = omega(f_rad_r[ii2], r[ii2], h_r[ii2], B_r[ii2])
    f_r = f(lw_r[ii2], dotm_r[ii2], r[ii2], omega_r[ii2])
tau_r = (20 * 2 ** 0.5 * dotm_r * f_r) / (3 * alpha * h_r ** 1.5 * r ** 1.5 * B_r ** 0.5 * f_rad_r ** 0.5)

# Here we calculate the spectrum of th accretion disk.
lamda = simpson(r * f_rad_r, r)
F_rad_r = (G_const * M * c_const * f_rad_r) / (Rg ** 2 * kappa)  # in erg/s/cm^2
T_r = (F_rad_r / sigma_const) ** 0.25  # in K
fcol = 2.3 - (1.3 * (1 + exp(-1))) / (1 + np.exp(k_const * T_r * 2.82 / (h_const * 5e15) - 1))  # color carrection
L_nu = np.zeros(num)
lnnu = np.linspace(log(1e14), log(1e18), num)
nu = np.exp(lnnu)  # in Hz
energy = (h_const * nu) / (1e3 * e_const * V_const)  # in keV
for i in range(num):
    L_nu[i] = ((8 * np.pi ** 2 * Rg ** 2 * h_const * nu[i] ** 3) / (c_const ** 2)) * simpson(
        r / (fcol ** 4 * np.exp((h_const * nu[i]) / (fcol * k_const * T_r)) - 1), r)
spec = L_nu * nu

# The function vertical_structure returns distribution over the non-uniform variation in height.
# Here we use the interpolation method to obtain the distribution over the height of the logarithmic change.
norm_z = np.logspace(-1, np.log10(r_out * 2 ** 0.5 / 2), num) # z increases logarithmically from 0.1 to √2/2 r_out.
position_rz = np.zeros(num)
vz_rz = np.zeros((num, num))
Rho_rz = np.zeros((num, num))
T_rz = np.zeros((num, num))
Pres_rz = np.zeros((num, num))
for ii3 in range(num):
    v_structure = vertical_structure(r[ii3], tau_r[ii3], f_rad_r[ii3])
    position_rz = v_structure[0] * r[ii3]
    F1 = interp1d(position_rz, v_structure[2] / r[ii3] ** 0.5, kind='slinear', fill_value="extrapolate")
    vz_rz[ii3] = F1(norm_z)
    if f_rad_r[ii3] < f_rad_crit(r[ii3]):
        F2 = interp1d(position_rz, (v_structure[3] * float(tau_r[ii3]) / (h_r[ii3] * r[ii3] * Rg * kappa)),
                      kind='slinear', bounds_error=False, fill_value=1e-20)
    else:
        F2 = interp1d(position_rz, (v_structure[3] * float(tau_r[ii3]) / (h_r[ii3] * r[ii3] * Rg * kappa)),
                      kind='slinear', fill_value="extrapolate")
    Rho_rz[ii3] = F2(norm_z)
    (Rho_rz[ii3])[(Rho_rz[ii3]) < 5e-14] = 5e-14
    F3 = interp1d(position_rz, v_structure[4], kind='slinear', fill_value="extrapolate")
    T_rz[ii3] = F3(norm_z)
    (T_rz[ii3])[(T_rz[ii3]) < 3] = 3
    F4 = interp1d(position_rz, v_structure[5], kind='slinear', fill_value="extrapolate")
    Pres_rz[ii3] = F4(norm_z)
    (Pres_rz[ii3])[(Pres_rz[ii3]) < 0.1] = 0.1

# Plot the structure of the accretion disk.
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(2, 2, 1)
im = ax1.imshow(Rho_rz.T, extent=[r[0], r[-1], norm_z[0], norm_z[-1]], origin='lower', cmap='Reds', norm=colors.LogNorm())
cbar = plt.colorbar(im, ax=ax1)
cbar.ax.set_title(r'$g/cm^3$')
ax1.axis('off')
ax1.set_title(r'Density distribution $\rho(r,z)$')

ax2 = fig.add_subplot(2, 2, 3)
ax2.loglog(r, F_rad_r,label=r'$f_{rad}$')
ax2.loglog(r, (G_const * M * c_const * f_rad_crit(r)) / (Rg ** 2 * kappa),label=r'$f_{rad}^{crit}$')
ax2.set_xlabel(r'$r/R_{g}$')
ax2.set_ylabel(r'$S/erg\cdot s^{-1} \cdot cm^{-2}$',color='r')
ax2.tick_params('y', colors='r')
ax2.legend(loc='center left')
ax2.set_title('Radiation flux and accretion rate distribution')
ax3 = ax2.twinx()
ax3.loglog(r,dotm_r,label=r'$\dot{M}$',color='g')
ax3.set_ylabel(r'$\dot{M}/{\dot{M}}_{Edd}$',color='b')
ax3.tick_params('y', colors='b')
ax3.legend(loc='center right')

ax4 = fig.add_subplot(1, 2, 2)
ax4.loglog(energy, L_nu * nu)
ax4.set_xlabel(r'$Energy /keV$')
ax4.set_ylabel(r'$\nu L_{\nu}/erg\cdot s^{-1}$')
ax4.set_title('The continuum spectra of the disks')
ax4.set_xlim(1e-4, 10)
ax4.set_ylim(1e43, 1e46)
plt.subplots_adjust(wspace=0.5, hspace=0.3)
plt.show()

# plt.imshow(T_rz.T, extent=[r[0], r[-1], norm_z[0], norm_z[-1]], origin='lower', cmap='Reds', norm=colors.LogNorm())
# cbar = plt.colorbar()
# cbar.ax.set_title(r'$K$')
# plt.axis('off')
# plt.title(r'Temperature distribution $T(r,z)$')
# plt.show()