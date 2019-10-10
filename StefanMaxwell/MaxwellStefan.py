########### Developed by Rafael Pereira do Carmo ############
## ATOMS - Applied Thermodynamics and Molecular Simulation ##
########### Federal University of Rio de Janeiro ############

#Transient unidimensional diffusion using Maxwell-Stefan model
import numpy as np
import matplotlib.pyplot as plt

#Example: ternary system (H2(1), N2(2) e CO2(3))
NC = 3
D = np.zeros([NC,NC])
D[1,0] = D[0,1] = 83.3 #mm²/s
D[2,0] = D[0,2] = 68.0 #mm²/s
D[2,1] = D[1,2] = 16.8 #mm²/s

L = 85.9 #mm
N = 200
MolFrac = np.zeros((3,N))
POS = np.linspace(0,L,N)
dPOS = POS[1]
dt = 0.001

#Initial condition
MolFrac[0,0:int(N/2)] = 0.48
MolFrac[1,0:int(N/2)] = 0.52
MolFrac[2,0:int(N/2)] = 0.0
MolFrac[0,int(N/2):] = 0.0
MolFrac[1,int(N/2):] = 0.48
MolFrac[2,int(N/2):] = 0.52

def DiffFlux(NC, D, MolFrac, dPOS, pos):
    NC_aux = NC - 1
    B = np.zeros([NC_aux,NC_aux])
    
    for i in range(0,NC_aux):
        B[i,i] = MolFrac[i,pos]/D[i,NC_aux]
        for j in range(0,i,1):
            B[i,i] += MolFrac[j,pos]/D[i,j]
            B[i,j] = MolFrac[i,pos]*(1/D[i,NC_aux]-1/D[i,j])
            B[j,i] = MolFrac[j,pos]*(1/D[j,NC_aux]-1/D[j,i])
        for j in range(i+1,NC-1,1):
            B[i,i] += MolFrac[j,pos]/D[i,j]
            B[i,j] = MolFrac[i,pos]*(1/D[i,NC_aux]-1/D[i,j])
            B[j,i] = MolFrac[j,pos]*(1/D[j,NC_aux]-1/D[j,i])
        B[i,i] += MolFrac[NC-1,pos]/D[i,NC-1]
    
    if(np.linalg.det(B) != 0):
        flux = np.linalg.inv(B) @ (MolFrac[:NC_aux,pos+1] - MolFrac[:NC_aux,pos])
    else:
        flux = np.zeros(NC-1)
    return -flux/dPOS


ts = 10000
for t in range(1,ts+1):
    #First point
    p = 0 #position = 1rst node
    J_prev = 0
    J = DiffFlux(NC,D,MolFrac,dPOS,p+1)
    MolFrac[:NC-1,p] -= dt*(J - J_prev)/dPOS
    MolFrac[NC-1,p] = 1 - np.sum(MolFrac[:NC-1,p])
    
    #Second point
    p = 1 #position = 2nd node
    MolFrac[:NC-1,p] -= dt*(J - J_prev)/dPOS
    MolFrac[NC-1,p] = 1 - np.sum(MolFrac[:NC-1,p])
    
    #Other points
    for p in range(2,N-1):
        J_prev = J.copy()
        J = DiffFlux(NC,D,MolFrac,dPOS,p)
        MolFrac[:NC-1,p] -= dt*(J - J_prev)/dPOS
        MolFrac[NC-1,p] = 1 - np.sum(MolFrac[:NC-1,p])
        
    #Last point
    p = N-1 #position = last node
    J_prev = J.copy()
    J = 0
    MolFrac[:NC-1,p] -= dt*(J - J_prev)/dPOS
    MolFrac[NC-1,p] = 1 - np.sum(MolFrac[:NC-1,p])
    
#Printing to file
fig1 = plt.figure()
ax1 = fig1.gca()
ax1.set_xlabel('Position')
ax1.set_ylabel('Mole fraction')
ax1.set_xlim(0,L)
ax1.set_ylim(0,0.8)
ax1.plot(POS,MolFrac[0,:],label=r'$H_2$')
ax1.plot(POS,MolFrac[1,:],label=r'$N_2$')
ax1.plot(POS,MolFrac[2,:],label=r'$CO_2$')
ax1.set_title(f'time = {(t*dt):-0.2F} s')
ax1.legend()
fig1.savefig('Diffusion_'+str(t)+'ts.png',dpi=200,bbox_inches='tight')

