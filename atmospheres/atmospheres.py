import numpy as np
import matplotlib.pyplot as plt
import dedalus2.public as d2

class DedalusAtmosphere():
    def __init__(self, z_basis, num_coeffs=20):

        self.num_coeffs = num_coeffs
        self.domain = d2.Domain([z_basis])
        
        self.z_atmosphere = z_basis.grid
        self.z = z_basis.grid

        self.fields = dict()
        self.keys = sorted(self.parameters.keys())
        for key in self.keys:
            print("setting {:s}".format(key))
            self.fields[key] = self.domain.new_field()
            self.fields[key]['g'] =  self.parameters[key]()

        self.N_parameters = len(self.keys)
        
        self.check_spectrum()
        print("Atmosphere spectrum has been output to: atmosphere_spectrum.png")
        
        self.truncate_atmosphere()

        self.check_atmosphere()
        print("Atmosphere comparison has been output to: atmosphere.png")
                        
    def truncate_atmosphere(self):
        for key in self.keys:
            self.fields[key]['c'][self.num_coeffs:] = 0

    def get_coeffs(self, key):
        return np.copy(self.fields[key]['c'][:self.num_coeffs])
    
    def get_values(self, key):
        return np.copy(self.fields[key]['g'].real)
        
    def check_spectrum(self):
        fig = plt.figure()
        N_plots = self.N_parameters
        
        for i_ax, key in enumerate(self.keys):
            ax = fig.add_subplot(1,N_plots,i_ax+1)

            ax.loglog(np.abs(self.fields[key]['c']*np.conj(self.fields[key]['c'])))
            
            ax.axvline(x=self.num_coeffs,linestyle='dashed')
            ax.set_title(self.titles[key])
            ax.set_xlabel(r"$T_n$")
            if i_ax == 0:
                ax.set_ylabel("power spectrum")

        fig.savefig("atmosphere_spectrum.png")
        plt.close(fig)

    def check_atmosphere(self):
        fig = plt.figure()
        N_plots = self.N_parameters
        
        for i_ax, key in enumerate(self.keys):
            ax1 = fig.add_subplot(2,N_plots,i_ax+1)

            ax1.plot(self.z, self.parameters[key]())
            ax1.plot(self.z, self.get_values(key))

            ax1.set_title(self.titles[key])
            
            ax2 = fig.add_subplot(2,N_plots,i_ax+1+N_plots)

            ax2.plot(self.z, np.abs(self.parameters[key]()/self.get_values(key)-1))

            ax2.set_xlabel(r"$z$")

        fig.savefig('atmosphere.png')
        plt.close(fig)
        
class Polytrope(DedalusAtmosphere):
    def __init__(self, gamma, polytropic_index, z0, z, **args):
        
        self.gamma = gamma
        
        self.polytropic_index = polytropic_index
        self.polytropic_index_ad = 1/(self.gamma-1)
        # shorthand versions
        self.n = self.polytropic_index
        self.n_ad = self.polytropic_index_ad

        self.epsilon = self.polytropic_index - self.polytropic_index_ad
        print("ε = ", self.epsilon)

        self.z0 = z0

        self.parameters = dict()
        self.titles = dict()
        
        self.parameters['grad_ln_rho'] = self.grad_ln_rho
        self.titles['grad_ln_rho'] = r"$\nabla \ln \rho_0$"
        self.parameters['grad_S'] = self.grad_S
        self.titles['grad_S'] = r"$\nabla S_0$"

        DedalusAtmosphere.__init__(self, z, **args)
                
    def grad_ln_rho(self):
        return -self.polytropic_index/(self.z0-self.z)

    def grad_S(self):
        entropy_gradient_factor = ((self.polytropic_index+1)/self.gamma - self.polytropic_index)
        print(entropy_gradient_factor)
        print(self.epsilon/self.gamma)
        return -entropy_gradient_factor/(self.z0-self.z)

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
    



if __name__ == "__main__":

    Lz = 10
    nz = 512*3/2
    
    z_basis = d2.Chebyshev(nz, interval=[0., Lz], dealias=2/3)

    gamma = 5/3
    epsilon = 1e-4
    poly_n = 1.5 - epsilon

    
    num_coeffs = 20

    atm = Polytrope(gamma, poly_n, Lz+1, z_basis, num_coeffs=num_coeffs)
