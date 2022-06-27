# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:03:09 2021

@author: Dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# Hashin-Shtrikman bounds formula for multimineral mixture 

def F_HS(K_minerals,G_minerals,mineral_fractions,porosity,fluid_K_modules,fluid_G_modules):   

    '''
This function computes Hashin-Shtrikman (HS) (also called Hashin-Shtrikman-
Walpole) bounds for multi component mixtures. The function takes into 
Bulk modules(K), shear modules(u), volume fraction of components 
in a mixture and the mixture porosity. The function returns Upper Hashin-
Shtrikman bulk modules (KHS_upper), Lower Hashin-Shtrikman bulk modules 
(KHS_lower), Upper Hashin-Shtrikman shear modules (uHS_upper) and lower 
Hashin-Shtrikman shear modules (uHS_lower).

For using this function, sequence of components in K_minerals, G_minerals, 
mineral_fractions arrays must be the same. 

The function input units:
    K_minerals - GPa (10e9 Pa);
    G_minerals - GPa (10e9 Pa);
    mineral_fractions - fraction;
    porosity - fraction;
    fluid_K_modules - GPa (10e9 Pa);
    fluid_G_modules - GPa (10e9 Pa);

The function output units:
    KHS_lower - GPa (10e9 Pa);
    KHS_upper - GPa (10e9 Pa);
    uHS_lower - GPa (10e9 Pa);
    uHS_upper - GPa (10e9 Pa).
    '''
    
    K_minerals=K_minerals.astype(float) # Convert argument values to float numbers
                                        # for correctly computation purposes      
    G_minerals=G_minerals.astype(float) # Convert argument values to float numbers
                                        # for correctly computation purposes
    if porosity>0:
        converter = (1-porosity)*mineral_fractions
        fraction = np.insert(converter, 0,porosity)
    else:
        fraction = mineral_fractions
            
    
    
    
    if porosity>0:
        K = np.array(np.insert(K_minerals, 0,fluid_K_modules))
    else:
        K = np.array(K_minerals)
    
    if porosity>0:    
        G = np.array(np.insert(G_minerals, 0,fluid_G_modules))
    else:
        G = np.array(G_minerals)

    
    K_sort=np.sort(K) #sort values in K array from small values to high values 
    G_sort=np.sort(G) #sort values in G array from small values to high values
    
    K_min=K_sort[0] #define K_min
    G_min=G_sort[0] #define u_min
    
    K_max=K_sort[-1] #define K_max
    G_max=G_sort[-1] #define u_max
    
    Stabilizator = 1e-100 #The value which is used only fro preventing division by zero 
    
    '''Calculate KHS_lower'''
    A=0
    for i in range(0,len(G)):        
        A=A+fraction[i]/(K[i]+(4/3)*G_min+Stabilizator)
    KHS_lower=1/(A+Stabilizator)-(4/3)*G_min
    
    '''Calculate KHS_upper'''
    A=0
    for i in range(0,len(G)):        
        A=A+fraction[i]/(K[i]+(4/3)*G_max+Stabilizator)
    KHS_upper=1/(A+Stabilizator)-(4/3)*G_max
    
    '''Calculate uHS_lower'''
    Ksi_min = (G_min/6)*((9*K_min+8*G_min)/(K_min+2*G_min+Stabilizator))
    A = 0
    for i in range (0,len(G)):
        A = A+fraction[i]/(G[i]+Ksi_min+Stabilizator)
    uHS_lower = 1/(A+Stabilizator)-Ksi_min    

    '''Calculate uHS_upper'''
    Ksi_max = (G_max/6)*((9*K_max+8*G_max)/(K_max+2*G_max+Stabilizator))
    A = 0
    for i in range (0,len(G)):
        A = A+fraction[i]/(G[i]+Ksi_max+Stabilizator)
    uHS_upper = 1/(A+Stabilizator)-Ksi_max    

    
    return KHS_lower, KHS_upper, uHS_lower, uHS_upper


Minerals_sequence = ['Olivine','Pyroxene','Plagioclase','Quartz','Orthoclase',\
                     'Muscovite','Biotite','Amphibole'] #Sequence of mineral values in lists below 


# SiO2 content in chemical classification model for igneous rocks, modified version of ”Physical
# Properties of Rocks” figure 1.2 (Sch¨on, 2015a)                             
Sio2_content = np.array([[40],[41],[42],[43],[44],[45],[46],[47],[48],[49],[50],\
                         [51],[52],[53],[54],[55],[56],[57],[58],[59],[60],[61],\
                             [62],[63],[64],[65],[66],[67],[68],[69],[70]]) 


# Mineral fraction on given SiO2 content
    
Olivine = [[1.00],[1.00],[0.96],[0.857],[0.722],[0.517],[0.372],[0.273],[0.196],[0.133],[0.0797],\
           [0.027],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],\
               [0],[0],[0],[0]] 
Pyroxene = [[0],[0],[0.04],[0.143],[0.278],[0.483],[0.628],[0.727],[0.787],[0.787],[0.743],\
            [0.68],[0.467],[0.277],[0.1527],[0.0727],[0.023],[0],[0],[0],[0],[0],\
                [0],[0],[0],[0],[0],[0],[0],[0],[0]]
Plagioclase = [[0],[0],[0],[0],[0],[0],[0],[0],[0.017],[0.08],[0.177],\
               [0.293],[0.427],[0.52],[0.59],[0.635],[0.647],[0.614],[0.5573],\
                   [0.501],[0.4467],[0.3867],[0.332],[0.2867],[0.2447],[0.213],\
                       [0.188],[0.1693],[0.162],[0.1586],[0.1646]]
Quartz = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0.0187],\
          [0.0567],[0.106],[0.146],[0.186],[0.226],[0.2587],[0.282],[0.2967],\
              [0.29867],[0.2893],[0.2753],[0.2513],[0.2267],[0.1973]]
Orthoclase = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],\
              [0],[0],[0.014],[0.031],[0.0567],[0.0847],[0.12],[0.16467],\
                  [0.20867],[0.2667],[0.32],[0.376],[0.4353],[0.4866]]
Muscovite = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],\
             [0.009],[0.014],[0.0213],[0.0233],[0.0307],[0.04467],[0.058],\
                 [0.0706],[0.08],[0.09],[0.092],[0.09],[0.0867],[0.0706],[0.0566]]
Biotite = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0.0233],\
           [0.033],[0.0447],[0.0493],[0.0587],[0.076],[0.092],[0.103],[0.108],\
               [0.11],[0.103],[0.0953],[0.0867],[0.07467],[0.066],[0.052],[0.0433],[0.0287]]
Amphibole = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0.1063],[0.1797],\
             [0.224],[0.233],[0.242],[0.2393],[0.2267],[0.2067],[0.19],[0.163],\
                 [0.1387],[0.122],[0.1053],[0.08867],[0.07467],[0.06333],[0.05867],\
                     [0.0533],[0.052]]



Chemical_class_model = np.array([Olivine,Pyroxene,Plagioclase,Quartz,Orthoclase,Muscovite,Biotite,Amphibole])  


# Bulk modules of minerals
K_minerals = np.array([129.8,102.7,71.5,37.2,46.8,52.1,50.6,87]) #GPa

# Shear modules of minerals
G_minerals = np.array([81.8,60.4,31.7,44.4,27.3,31.6,27.3, 43.0]) #GPa 

# Density of minerals    
Density_minerals = np.array([[3321.0],[3285.0],[2663.0],[2650.0],[2570.0],[2790.0],[3050.0],[3124.0]]) #kg/m3

# Empty lists for using in the loop below
KHS_upper = []
KHS_lower = []
uHS_lower = []
uHS_upper = []
Rock_density = []

# The loop for implementing HS bounds for mineral mixtures on different Sio2 
# contents and for calculation densities of those mixtures 
for i in range(0,len(Sio2_content)):
    Mineral_fractions_on_SIO2_content=Chemical_class_model[:,i]
    HS = F_HS(K_minerals,G_minerals,Mineral_fractions_on_SIO2_content,0.0,0.0,0.0)   
    KHS_lower.append(HS[0])
    KHS_upper.append(HS[1])
    uHS_lower.append(HS[2])
    uHS_upper.append(HS[3])
    
    Rock_density_0 = (Density_minerals*Mineral_fractions_on_SIO2_content)
    Rock_density_1 = np.sum(Rock_density_0)
    Rock_density.append(Rock_density_1)

# Empty lists for using in the loop below
Vp_HS_upper = np.zeros(len(KHS_upper))
Vp_HS_lower = np.zeros(len(KHS_lower))

Vs_HS_lower = np.zeros(len(uHS_lower))
Vs_HS_upper = np.zeros(len(uHS_upper))

VpVs_HS_lower = np.zeros(len(uHS_lower))
VpVs_HS_upper = np.zeros(len(uHS_upper))

AI_HS_lower = np.zeros(len(uHS_lower))
AI_HS_upper = np.zeros(len(uHS_upper))

for i in range(0,len(KHS_lower)):
    Vp_HS_lower[i] = (np.sqrt((KHS_lower[i]*10**9 + (4/3)*(uHS_lower[i]*10**9))/Rock_density[i]))/1000 # km/s
    Vp_HS_upper[i] = (np.sqrt((KHS_upper[i]*10**9 + (4/3)*(uHS_upper[i]*10**9))/Rock_density[i]))/1000 # km/s
    
    Vs_HS_upper[i] = (np.sqrt((uHS_upper[i]*10**9)/Rock_density[i]))/1000 # km/s
    Vs_HS_lower[i] = (np.sqrt((uHS_lower[i]*10**9)/Rock_density[i]))/1000 # km/s
    
    VpVs_HS_lower[i] =Vp_HS_lower[i]/Vs_HS_lower[i]
    VpVs_HS_upper[i] =Vp_HS_upper[i]/Vs_HS_upper[i]
    
    AI_HS_lower[i] = Vp_HS_lower[i]*1000*Rock_density[i] #[m/s]*[kg/m3]
    AI_HS_upper[i] = Vp_HS_lower[i]*1000*Rock_density[i] #[m/s]*[kg/m3]
    
    
##############################################################################
#For plotting data from Bent Hill area

Vp_sul_pyrite_dom = np.array([4.7873,	4.7873,	4.7873,	4.9210,	4.1863,	4.1863,\
                              100,	4.7133,	4.7133,	4.7133,	5.9160,	5.9160,	4.8730,\
                                  730,	4.8730,	5.8547,	5.8547,	5.8547,	5.8547,	5.8547,\
                                      500,	5.0500,	4.9563,	4.9563,	4.9563,	4.9563, 5.9400,\
                                          400,	4.3320,	6.0000,	6.0000,	4.3033,	4.3033,	4.0983,\
                                              983,	3.3080,	5.1457,	5.1457,	3.5330,	3.5330]) #km/s
    
Dens_sul_pyrite_dom = np.array([4190,4190,4190,4200,3860,3860,3970,3610,3610,\
                                3610,3920,3920,3510,3510,3510,4300,4300,4300,\
                                    4300,4300,4300,4300,3920,3920,3920,3920,\
                                        4090,4090,4090,4090,4160,3740,3740,\
                                            3850,3850,3680,4050,4050,3530,3530]) #kg/m3

##############################################################################
# For plotting Sulphite rocks

K_Pyrite = 147.4 #GPa
u_Pyrite = 132.5 #GPa
Density_Pyrite = 4930 #kg/m3  
 
K_Pyrrohotite = 53.8 #GPa
u_Pyrrohotite = 34.7 #GPa
Density_Pyrrohotite = 4550 #kg/m3  


K_Sphalerite = 53.8 #GPa
u_Sphalerite = 34.7 #GPa
Density_Sphalerite = 4080 #kg/m3  


Vp_Pyrite = (np.sqrt((K_Pyrite*10**9 + (4/3)*(u_Pyrite*10**9))/Density_Pyrite))/1000 # km/s
Vp_Pyrrohotite = (np.sqrt((K_Pyrrohotite*10**9 + (4/3)*(u_Pyrrohotite*10**9))/Density_Pyrrohotite))/1000 # km/s
Vp_Sphalerite = (np.sqrt((K_Sphalerite*10**9 + (4/3)*(u_Sphalerite*10**9))/Density_Sphalerite))/1000 # km/s
    
Vs_Pyrite = (np.sqrt((u_Pyrite*10**9)/Density_Pyrite))/1000 # km/s
Vs_Pyrrohotite = (np.sqrt((u_Pyrrohotite*10**9)/Density_Pyrrohotite))/1000 # km/s
Vs_Sphalerite = (np.sqrt((u_Sphalerite*10**9)/Density_Sphalerite))/1000 # km/s
    
VpVs_Pyrite = Vp_Pyrite/Vs_Pyrite
VpVs_Pyrrohotite = Vp_Pyrrohotite/Vs_Pyrrohotite
VpVs_Sphalerite =Vp_Sphalerite/Vs_Sphalerite
    
AI_Pyrite = 1000*Vp_Pyrite*Density_Pyrite #[m/s]*[kg/m3]
AI_Pyrrohotite = 1000*Vp_Pyrrohotite*Density_Pyrrohotite #[m/s]*[kg/m3]
AI_Sphalerite = 1000*Vp_Sphalerite*Density_Sphalerite #[m/s]*[kg/m3]

Sulfide_fractions = np.array([[0,1],[0.1,0.9],[0.2,0.8],[0.3,0.7],[0.4,0.6],[0.5,0.5],[0.6,0.4],[0.7,0.3],[0.8,0.2],[0.9,0.1],[1,0]])
K_Pyrite_host_rock_HS_lower = []
K_Pyrite_host_rock_HS_upper = []
u_Pyrite_host_rock_HS_lower = []
u_Pyrite_host_rock_HS_upper = []
Density_Pyrite_host_rock = []

# The loop for implementing HS bounds for mixtures of sulfide and host rock 
#  and for calculation densities of those mixtures 
for i in range(0,len(Sulfide_fractions)):
    K_Pyrite_host_rock = np.array([K_Pyrite,KHS_lower[-1]])
    u_Pyrite_host_rock = np.array([u_Pyrite,uHS_lower[-1]])
    density_Pyrite_host_rock = np.array([Density_Pyrite,Rock_density[-1]])
    
    HS_Pyrite = F_HS(K_Pyrite_host_rock,u_Pyrite_host_rock,Sulfide_fractions[i],0.0,0.0,0.0)   
    
    K_Pyrite_host_rock_HS_lower.append(HS_Pyrite[0])
    K_Pyrite_host_rock_HS_upper.append(HS_Pyrite[1])
    u_Pyrite_host_rock_HS_lower.append(HS_Pyrite[2])
    u_Pyrite_host_rock_HS_upper.append(HS_Pyrite[3])
    
    Rock_density_3 = (density_Pyrite_host_rock*Sulfide_fractions[i])
    Rock_density_4 = np.sum(Rock_density_3)
    Density_Pyrite_host_rock.append(Rock_density_4)


# Empty lists for using in the loop below
Vp_Pyrite_host_rock_HS_upper = np.zeros(len(Sulfide_fractions))
Vp_Pyrite_host_rock_HS_lower = np.zeros(len(Sulfide_fractions))

Vs_Pyrite_host_rock_HS_lower = np.zeros(len(Sulfide_fractions))
Vs_Pyrite_host_rock_HS_upper = np.zeros(len(Sulfide_fractions))

VpVs_Pyrite_host_rock_HS_lower = np.zeros(len(Sulfide_fractions))
VpVs_Pyrite_host_rock_HS_upper = np.zeros(len(Sulfide_fractions))

AI_Pyrite_host_rock_HS_lower = np.zeros(len(Sulfide_fractions))
AI_Pyrite_host_rock_HS_upper = np.zeros(len(Sulfide_fractions))

for i in range(0,len(Sulfide_fractions)):
    Vp_Pyrite_host_rock_HS_lower[i] = (np.sqrt((K_Pyrite_host_rock_HS_lower[i]*10**9 + (4/3)*(u_Pyrite_host_rock_HS_lower[i]*10**9))/Density_Pyrite_host_rock[i]))/1000 # km/s
    Vp_Pyrite_host_rock_HS_upper[i] = (np.sqrt((K_Pyrite_host_rock_HS_upper[i]*10**9 + (4/3)*(u_Pyrite_host_rock_HS_upper[i]*10**9))/Density_Pyrite_host_rock[i]))/1000 # km/s
    
    Vs_Pyrite_host_rock_HS_upper[i] = (np.sqrt((u_Pyrite_host_rock_HS_upper[i]*10**9)/Density_Pyrite_host_rock[i]))/1000 # km/s
    Vs_Pyrite_host_rock_HS_lower[i] = (np.sqrt((u_Pyrite_host_rock_HS_lower[i]*10**9)/Density_Pyrite_host_rock[i]))/1000 # km/s
    
    VpVs_Pyrite_host_rock_HS_lower[i] =Vp_Pyrite_host_rock_HS_lower[i]/Vs_Pyrite_host_rock_HS_lower[i]
    VpVs_Pyrite_host_rock_HS_upper[i] =Vp_Pyrite_host_rock_HS_upper[i]/Vs_Pyrite_host_rock_HS_upper[i]
    
    AI_Pyrite_host_rock_HS_lower[i] = 1000*Vp_Pyrite_host_rock_HS_lower[i]*Density_Pyrite_host_rock[i] #[m/s]*[kg/m3]
    AI_Pyrite_host_rock_HS_upper[i] = 1000*Vp_Pyrite_host_rock_HS_upper[i]*Density_Pyrite_host_rock[i] #[m/s]*[kg/m3]
    

##############################################################################
#Plotting host rock Vp/Vs and AI calculated by HS on Salisburies plot

ax2 = plt.figure(3)
#Plotting data from the paper
#plt.xlim(2.0,5.25)
#plt.ylim(3.5,9.0) 



plt.plot(AI_HS_lower[0:5], VpVs_HS_lower[0:5],'v',markersize=0.3,c='darkred',label='VpVS_HS_lower_Ultramafic'.format('r'))
plt.plot(AI_HS_upper[0:5], VpVs_HS_upper[0:5],'^',markersize=0.3,c='darkred',label='VpVs_HS_upper_Ultramafic'.format('r'))

plt.plot(AI_HS_lower[5:12], VpVs_HS_lower[5:12],'v',markersize=0.3,c='red',label='VpVs_HS_lower_Mafic'.format('r'))
plt.plot(AI_HS_upper[5:12], VpVs_HS_upper[5:12],'^',markersize=0.3,c='red',label='VpVs_HS_upper_Mafic'.format('r'))

plt.plot(AI_HS_lower[12:23], VpVs_HS_lower[12:23],'v',markersize=0.3,c='green',label='VpVs_HS_lower_Intermediate'.format('r'))
plt.plot(AI_HS_upper[12:23], VpVs_HS_upper[12:23],'^',markersize=0.3,c='green',label='VpVs_HS_upper_Intermediate'.format('r'))

plt.plot(AI_HS_lower[23:31], VpVs_HS_lower[23:31],'v',markersize=0.3,c='gold',label='VpVs_HS_lower_Felsic'.format('r'))
plt.plot(AI_HS_upper[23:31], VpVs_HS_upper[23:31],'^',markersize=0.3,c='gold',label='VpVs_HS_upper_Felsic'.format('r'))

plt.plot(AI_Pyrite, VpVs_Pyrite,'*',markersize=1,c='red',label='VpVs_Pyrite'.format('r'))
plt.plot(AI_Pyrrohotite, VpVs_Pyrrohotite,'*',markersize=1,c='blue',label='VpVs_Pyrrohotite'.format('r'))
plt.plot(AI_Sphalerite, VpVs_Sphalerite,'*',markersize=1,c='black',label='VpVs_Sphalerite'.format('r'))




plt.plot(AI_Pyrite_host_rock_HS_lower, VpVs_Pyrite_host_rock_HS_lower,'+',markersize=0.3,c='green',label='VpVS_HS_lower_Pyrite_host_rock'.format('r'))
plt.plot(AI_Pyrite_host_rock_HS_upper, VpVs_Pyrite_host_rock_HS_lower,'+',markersize=0.3,c='yellow',label='VpVs_HS_upper_Pyrite_host_rock'.format('r'))


#Plot settings
plt.legend(fontsize =3)
plt.xlabel('Acoustic Impedence, [rayls]')
plt.ylabel('Vp/Vs, [-]')
plt.title('Hashin-Shtrikman Vp/Vs and Acoustic Impedence for 0% porisity Host rocks')


#Saving resulted image for further optimization
plt.savefig('Image_reading/Host_rock_sulfides_VpVs_AI_using_HS_eq.jpg', dpi=500, bbox_inches='tight')

plt.show()


