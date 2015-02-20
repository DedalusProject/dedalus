import numpy as np
import matplotlib.pyplot as plt
from .. import public as d2


class DedalusAtmosphere():
    def __init__(self, z_basis, num_coeffs=20):

        self.num_coeffs = num_coeffs

        self.domain = d2.Domain([z_basis])
        self.z = z_basis.grid

        self.atmosphere = dict()
        self.keys = sorted(self.base_atmosphere.keys())
        self.N_parameters = len(self.keys)

        self.work = self.domain.new_field()

        for key in self.keys:
            print("setting {:s}".format(key))
            self.atmosphere[key] = self.domain.new_field()
            self.atmosphere[key]['g'] =  self.base_atmosphere[key]

            self.check_spectrum(key)
            self.truncate_atmosphere(key)
            self.check_atmosphere(key)


    def truncate_atmosphere(self, key):
        self.atmosphere[key]['c'][self.num_coeffs:] = 0

    def get_coeffs(self, key):
        return np.copy(self.atmosphere[key]['c'][:self.num_coeffs])

    def get_values(self, key):
        return np.copy(self.atmosphere[key]['g'].real)


    def check_spectrum(self, key, individual_plots=True):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        power_spectrum = np.sqrt(np.abs(self.atmosphere[key]['c']*np.conj(self.atmosphere[key]['c'])))
        ax.loglog(power_spectrum)

        ax.axvline(x=self.num_coeffs,linestyle='dashed')
        ax.set_title(self.titles[key])
        ax.set_xlabel(r"$T_n$")

        sp_bound = np.max(power_spectrum)*(1e-8)
        dp_bound = np.max(power_spectrum)*(1e-16)

        ax.axhline(y=sp_bound,linestyle='dashed')
        ax.axhline(y=dp_bound,linestyle='dashed')


        ax.set_ylabel("amplitude from power spectrum")

        atm_file = "spectrum_{:s}.png".format(key)
        fig.savefig(atm_file)
        plt.close(fig)
        print("Atmosphere spectrum has been output to: {:s}".format(atm_file))

    def check_atmosphere(self, key, individual_plots=True):
        fig = plt.figure()
        N_plots = 2
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)

        ax1.plot(self.z, self.base_atmosphere[key])
        ax1.plot(self.z, self.get_values(key))

        ax1.set_ylabel(self.titles[key])
        ax1.set_title("truncated atmosphere {:s}".format(self.titles[key]))

        ax2.semilogy(self.z, np.abs(self.base_atmosphere[key]/self.get_values(key)-1))

        ax2.set_ylabel("relative error")
        ax2.set_xlabel(r"$z$")

        atm_file = 'values_{:s}.png'.format(key)
        fig.savefig(atm_file)
        plt.close(fig)
        print("Atmosphere comparison has been output to: {:s}".format(atm_file))



class Polytrope(DedalusAtmosphere):
    def __init__(self, gamma, polytropic_index, z0, z, **args):
        self.name = "polytrope"

        self.gamma = gamma
        self.polytropic_index = polytropic_index
        self.polytropic_index_ad = 1/(self.gamma-1)
        # shorthand versions
        self.n = self.polytropic_index
        self.n_ad = self.polytropic_index_ad

        self.epsilon = self.polytropic_index_ad - self.polytropic_index
        print("ε = ", self.epsilon)
        print("Nρ = ", self.polytropic_index*np.log(z0))


        #self.entropy_gradient_factor = ((self.polytropic_index+1)/self.gamma - self.polytropic_index)
        #print(entropy_gradient_factor)
        #print(self.epsilon/self.gamma)
        self.entropy_gradient_factor = self.epsilon/self.gamma

        self.z0 = z0
        self.z = z.grid

        self.base_atmosphere = dict()
        self.titles = dict()

        self.base_atmosphere['grad_ln_rho'] = self.grad_ln_rho()
        self.titles['grad_ln_rho'] = r"$\nabla \ln \rho_0$"
        self.base_atmosphere['grad_S'] = self.grad_S()
        self.titles['grad_S'] = r"$\nabla S_0$"
        self.base_atmosphere['grad_ln_T'] = self.grad_ln_T()
        self.titles['grad_ln_T'] = r"$\nabla \ln T_0$"
        self.base_atmosphere['rho'] = self.rho()
        self.titles['rho'] = r"$\rho_0$"
        self.base_atmosphere['T'] = self.T()
        self.titles['T'] = r"$T_0$"
        self.base_atmosphere['P'] = self.P()
        self.titles['P'] = r"$P_0$"


        DedalusAtmosphere.__init__(self, z, **args)

    def grad_ln_rho(self):
        return -self.polytropic_index/(self.z0-self.z)

    def grad_S(self):
        return -self.entropy_gradient_factor/(self.z0-self.z)

    def grad_ln_T(self):
        return -1/(self.z0-self.z)

    def rho(self):
        return (self.z0-self.z)**self.polytropic_index

    def T(self):
        return (self.z0-self.z)

    def P(self):
        return (self.z0-self.z)**(self.polytropic_index+1)

    def S(self):
        return


class ScaledPolytrope(Polytrope):
    def __init__(self, gamma, polytropic_index, z0, z, **args):
        Polytrope.__init__(self, gamma, polytropic_index, z0, z, **args)
        self.scale_factor = (self.z0-self.z)
        self.name = "scaled polytrope"

    def grad_ln_rho(self):
        return -self.polytropic_index*np.ones_like(self.z)

    def grad_S(self):
        return -self.entropy_gradient_factor*np.ones_like(self.z)

    def grad_ln_T(self):
        return -np.ones_like(self.z)

    def rho(self):
        return (self.z0-self.z)**(self.polytropic_index-1)

    def T(self):
        return np.ones_like(self.z)

    def P(self):
        return (self.z0-self.z)**(self.polytropic_index)

    def S(self):
        return



if __name__ == "__main__":

    Lz = 10
    Lz = 106
    Lz = 740

    nz = 1024*3/2

    z_basis = d2.Chebyshev(nz, interval=[0., Lz], dealias=2/3)

    gamma = 5/3
    epsilon = 1e-4
    poly_n = 1.5 - epsilon


    num_coeffs = 50

    atm = ScaledPolytrope(gamma, poly_n, Lz+1, z_basis, num_coeffs=num_coeffs)
    #atm = Polytrope(gamma, poly_n, Lz+1, z_basis, num_coeffs=num_coeffs)
