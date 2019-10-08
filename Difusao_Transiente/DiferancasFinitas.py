
import numpy as np
import matplotlib.pyplot as plt


def AnaliticSol(position,C1,C2,L,DAB,t):
    series = np.zeros(position.size)
    for n in range(1,40):
        series += np.sin(n*np.pi*position/L)*np.exp(-DAB*t*(n*np.pi/L)**2)/n
    return (C2-C1)*(position/L + 2*series/np.pi) + C1

def FiniteDifMet(N,Cas,Ca0,L,DAB,t,dt):
    dx = L/(N+1)
    x = np.linspace(0,L,N+2)
    CA = np.zeros(N) + Ca0
    CA_aux = np.zeros(N+2)
    CA_aux[0] = Cas
    CA_aux[N+1] = Ca0

    A = np.zeros([N,N])
    A[0,0] = (1 - 2*DAB*dt/dx**2)
    A[0,1] = DAB*dt/dx**2
    for i in range(1,N-1):
        A[i,i-1] = DAB*dt/dx**2
        A[i,i] = (1 - 2*DAB*dt/dx**2)
        A[i,i+1] = DAB*dt/dx**2
    A[N-1,N-2] = DAB*dt/dx**2
    A[N-1,N-1] = (1 - 2*DAB*dt/dx**2)

    b = np.zeros(N)
    b[0] = DAB*dt/dx**2*Cas
    b[N-1] = DAB*dt/dx**2*Ca0

    t_aux = 0
    while(t_aux < t):
        t_aux += dt
        CA = (A @ CA)+ b
        
    CA_aux[1:N+1] = CA
    return x, CA_aux

Cas = 1
Ca0 = 0
L = 1
DAB = 0.001
t = 4

fig1 = plt.figure(1)
ax1 = fig1.gca()
dt = 0.02
N = 5
for i in range(5):
    x,CA = FiniteDifMet(N,Cas,Ca0,L,DAB,t,dt)
    ax1.plot(x,CA,lw=2,label='FDM'+str(N))
    N *= 2

N_an = 100
x_an = np.linspace(0,L,N_an)
CA_an = AnaliticSol(x_an,Cas,Ca0,L,DAB,t)
ax1.plot(x_an,CA_an,'-k',lw=2,label='Analitic')
ax1.legend(fontsize=14)
ax1.set_xlim(0,L/2)
ax1.set_ylim(Ca0,Cas)
ax1.set_xlabel('x',fontsize=14)
ax1.set_ylabel('T*',fontsize=14)
ax1.tick_params(labelsize=14)
plt.savefig('ConvergenciaMalha.png',dpi=400,bbox_inches='tight')
plt.show()




