import numpy as np
import matplotlib.pyplot as plt
import dedalus2.public as d2

class polytrope():#dedalus_atmosphere):
    def __init__(self, gamma, polytropic_index, z0, z):
        #dedalus_atmosphere.__init__(**args)
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
        self.parameters['grad_ln_rho'] = self.grad_ln_rho
        self.parameters['grad_S'] = self.grad_S

        self.set_grid(z.grid)
        
    def set_grid(self, z):
        self.z = z
        
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
    
class dedalus_atmosphere():
    def __init__(self, z_basis, atmosphere, num_coeffs=20):
        self.atmosphere = atmosphere
        self.num_coeffs = num_coeffs
        self.domain = d2.Domain([z_basis])
        
        self.z_atmosphere = z_basis.grid
        self.z = z_basis.grid

        self.grad_ln_rho_atmosphere = self.domain.new_field()
        self.grad_S_atmosphere = self.domain.new_field()

        self.atmosphere.set_grid(self.z_atmosphere)
        
        self.set_atmosphere()

        self.check_spectrum()
        print("Atmosphere spectrum has been output to: atmosphere_spectrum.png")
        
        self.truncate_atmosphere()

        self.check_atmosphere()
        print("Atmosphere comparison has been output to: atmosphere.png")
        
    def set_atmosphere(self):
        self.grad_ln_rho_atmosphere['g'] = self.atmosphere.parameters['grad_ln_rho']()
        self.grad_S_atmosphere['g'] = self.atmosphere.parameters['grad_S']()
                
    def truncate_atmosphere(self):
        self.grad_ln_rho_atmosphere['c'][self.num_coeffs:] = 0
        self.grad_S_atmosphere['c'][self.num_coeffs:] = 0

    def grad_ln_rho_coeffs(self):
        return np.copy(self.grad_ln_rho_atmosphere['c'][:self.num_coeffs])
    
    def grad_S_coeffs(self):
        return np.copy(self.grad_S_atmosphere['c'][:self.num_coeffs])

    def grad_ln_rho(self):
        return np.copy(self.grad_ln_rho_atmosphere['g'].real)
    
    def grad_S(self):
        return np.copy(self.grad_S_atmosphere['g'].real)
    
    def check_spectrum(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax1.loglog(np.abs(self.grad_ln_rho_atmosphere['c']*np.conj(self.grad_ln_rho_atmosphere['c'])))
        ax1.axvline(x=self.num_coeffs,linestyle='dashed')
        ax1.set_title(r"$\nabla \ln \rho_0$")
        ax1.set_xlabel(r"$T_n$")
        ax1.set_ylabel("power spectrum")
        ax2 = fig.add_subplot(1,2,2)
        ax2.loglog(np.abs(self.grad_S_atmosphere['c']*np.conj(self.grad_S_atmosphere['c'])))
        ax2.axvline(x=self.num_coeffs, linestyle='dashed')
        ax2.set_title(r"$\nabla S_0$")
        ax2.set_xlabel(r"$T_n$")
        fig.savefig("atmosphere_spectrum.png")
        plt.close(fig)

    def check_atmosphere(self):
        fig = plt.figure()
        N_atm_q = 2

        ax1 = fig.add_subplot(2,N_atm_q,1)
        ax1.plot(self.z_atmosphere, self.grad_ln_rho())
        ax1.plot(self.z_atmosphere, self.atmosphere.parameters['grad_ln_rho']())
        ax1.set_title(r"$\nabla \ln \rho_0$")
        ax3 = fig.add_subplot(2,N_atm_q,3)
        ax3.plot(self.z_atmosphere, 
                 np.abs(self.grad_ln_rho()/self.atmosphere.parameters['grad_ln_rho']()-1))

        ax2 = fig.add_subplot(2,N_atm_q,2)
        ax2.plot(self.z_atmosphere, self.grad_S())
        ax2.plot(self.z_atmosphere, self.atmosphere.parameters['grad_S']())
        ax2.set_title(r"$\nabla S_0$")
        ax4 = fig.add_subplot(2,N_atm_q,4)
        ax4.plot(self.z_atmosphere, 
                 np.abs(self.grad_S()/self.atmosphere.parameters['grad_S']()-1))
        fig.savefig('atmosphere.png')
        plt.close(fig)


if __name__ == "__main__":

    Lz = 10
    nz = 512*3/2
    
    z_basis = d2.Chebyshev(nz, interval=[0., Lz], dealias=2/3)

    gamma = 5/3
    epsilon = 1e-4
    poly_n = 1.5 - epsilon

    
    num_coeffs = 20

    atm = polytrope(gamma, poly_n, Lz+1, z_basis)

    dedalus_atm = dedalus_atmosphere(z_basis, atm, num_coeffs=num_coeffs)
