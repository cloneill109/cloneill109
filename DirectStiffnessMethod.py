# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:12:50 2022

@author: Connor's Laptop
"""
#Clearing previous
# Imports

import copy #Allows to create copies of objects in memory
import math #Math funcitonality
import numpy as np #Numpy for working with arrays
import matplotlib.pyplot as plt #Plotting functionality
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D #3D plotting functionality
import pandas as pd


#===================Start of Data Entry======================
#Constants
E = 200*10**9  #(N/m^2)
r = 0.005   #(m)
A = .0000785      #(m^2)
xFac = 100        #Scale factor for plotted displacements
y_stress = 95*10**6  #Yield Stress
#Nodal coordinates [x, y, z] (in ascending node order)
d1=1.7/7
d2=1.38/7
d3=1.26/7
d4=.9/7
z2=.85
z3=z2+.24
z4=z3+.5
z5=z4+.44
z6=z5+.78
z7=z6+.3
z8=z7+.47
alter1=.1
alter2=.2
alter3=.4
#Landing leg swivel
zs=.25
ys=5*d1
xs=.25

nodes = np.array([[0*d1,3*d1,0],
                  [1*d1,2*d1,0],
                  [2*d1,1*d1,0],
                  [3*d1,0*d1,0],
                  [4*d1,0*d1,0],
                  [5*d1,1*d1,0],
                  [6*d1,2*d1,0],
                  [7*d1,3*d1,0],
                  [7*d1,4*d1,0],
                  [6*d1,5*d1,0],
                  [5*d1,6*d1,0],
                  [4*d1,7*d1,0],
                  [3*d1,7*d1,0],
                  [2*d1,6*d1,0],
                  [1*d1,5*d1,0],
                  [0*d1,4*d1,0],
                  
                  [0*d1,3*d1,z2],
                  [1*d1,2*d1,z2],
                  [2*d1,1*d1,z2],
                  [3*d1,0*d1,z2],
                  [4*d1,0*d1,z2],
                  [5*d1,1*d1,z2],
                  [6*d1,2*d1,z2],
                  [7*d1,3*d1,z2],
                  [7*d1,4*d1,z2],
                  [6*d1,5*d1,z2],
                  [5*d1,6*d1,z2],
                  [4*d1,7*d1,z2],
                  [3*d1,7*d1,z2],
                  [2*d1,6*d1,z2],
                  [1*d1,5*d1,z2],
                  [0*d1,4*d1,z2],
                  
                  [0*d1,3*d1,z3],
                  [1*d1,2*d1,z3],
                  [2*d1,1*d1,z3],
                  [3*d1,0*d1,z3],
                  [4*d1,0*d1,z3],
                  [5*d1,1*d1,z3],
                  [6*d1,2*d1,z3],
                  [7*d1,3*d1,z3],
                  [7*d1,4*d1,z3],
                  [6*d1,5*d1,z3],
                  [5*d1,6*d1,z3],
                  [4*d1,7*d1,z3],
                  [3*d1,7*d1,z3],
                  [2*d1,6*d1,z3],
                  [1*d1,5*d1,z3],
                  [0*d1,4*d1,z3],
                  
                  [0*d1,3*d1,z4],
                  [1*d1,2*d1,z4],
                  [2*d1,1*d1,z4],
                  [3*d1,0*d1,z4],
                  [4*d1,0*d1,z4],
                  [5*d1,1*d1,z4],
                  [6*d1,2*d1,z4],
                  [7*d1,3*d1,z4],
                  [7*d1,4*d1,z4],
                  [0*d1,4*d1,z4],
                  
                  [0*d1,3*d1,z5],
                  [1*d1,2*d1,z5],
                  [2*d1,1*d1,z5],
                  [3*d1,0*d1,z5],
                  [4*d1,0*d1,z5],
                  [5*d1,1*d1,z5],
                  [6*d1,2*d1,z5],
                  [7*d1,3*d1,z5],
                  [7*d1,4*d1,z5],
                  [0*d1,4*d1,z5],
                  
                  [alter1+0*d2,alter1+3*d2,z6],
                  [alter1+1*d2,alter1+2*d2,z6],
                  [alter1+2*d2,alter1+1*d2,z6],
                  [alter1+3*d2,alter1+0*d2,z6],
                  [alter1+4*d2,alter1+0*d2,z6],
                  [alter1+5*d2,alter1+1*d2,z6],
                  [alter1+6*d2,alter1+2*d2,z6],
                  [alter1+7*d2,alter1+3*d2,z6],
                  [alter1+7*d2,alter1+4*d2,z6],
                  [alter1+0*d2,alter1+4*d2,z6],
                  
                  [alter2+0*d3,alter2+3*d3,z7],
                  [alter2+1*d3,alter2+2*d3,z7],
                  [alter2+2*d3,alter2+1*d3,z7],
                  [alter2+3*d3,alter2+0*d3,z7],
                  [alter2+4*d3,alter2+0*d3,z7],
                  [alter2+5*d3,alter2+1*d3,z7],
                  [alter2+6*d3,alter2+2*d3,z7],
                  [alter2+7*d3,alter2+3*d3,z7],
                  [alter2+7*d3,alter2+4*d3,z7],
                  [alter2+6*d3,alter2+5*d3,z7],
                  [alter2+5*d3,alter2+6*d3,z7],
                  [alter2+4*d3,alter2+7*d3,z7],
                  [alter2+3*d3,alter2+7*d3,z7],
                  [alter2+2*d3,alter2+6*d3,z7],
                  [alter2+1*d3,alter2+5*d3,z7],
                  [alter2+0*d3,alter2+4*d3,z7],
                  
                  [alter3+0*d4,alter3+3*d4,z8],
                  [alter3+1*d4,alter3+2*d4,z8],
                  [alter3+2*d4,alter3+1*d4,z8],
                  [alter3+3*d4,alter3+0*d4,z8],
                  [alter3+4*d4,alter3+0*d4,z8],
                  [alter3+5*d4,alter3+1*d4,z8],
                  [alter3+6*d4,alter3+2*d4,z8],
                  [alter3+7*d4,alter3+3*d4,z8],
                  [alter3+7*d4,alter3+4*d4,z8],
                  [alter3+6*d4,alter3+5*d4,z8],
                  [alter3+5*d4,alter3+6*d4,z8],
                  [alter3+4*d4,alter3+7*d4,z8],
                  [alter3+3*d4,alter3+7*d4,z8],
                  [alter3+2*d4,alter3+6*d4,z8],
                  [alter3+1*d4,alter3+5*d4,z8],
                  [alter3+0*d4,alter3+4*d4,z8],
                  
                  #Landing Legs
                   [7*d1+xs,4*d1+ys,z3-zs],
                   [0*d1-xs,4*d1+ys,z3-zs],
                   [0*d1-xs,4*d1+ys,z6+zs],
                   [7*d1+xs,4*d1+ys,z6+zs]
                ])

#Members [node_i, node_j, node_k]
members = np.array([[1,2],
                   [2,3],
                   [3,4],
                   [4,5],
                   [5,6],
                   [6,7],
                   [7,8],
                   [8,9],
                   [9,10],
                   [10,11],
                   [11,12],
                   [12,13],
                   [13,14],
                   [14,15],
                   [15,16],
                   [16,1],
                   
                   #Connecting L1 and 2
                   [1,18],
                   [3,18],
                   [3,20],
                   [5,20],
                   [5,22],
                   [7,22],
                   [7,24],
                   [9,24],
                   [9,26],
                   [11,26],
                   [11,28],
                   [13,28],
                   [13,30],
                   [15,30],
                   [15,32],
                   [1,32],
                   [1,17],
                   [2,18],
                   [3,19],
                   [4,20],
                   [5,21],
                   [6,22],
                   [7,23],
                   [8,24],
                   [9,25],
                   [10,26],
                   [11,27],
                   [12,28],
                   [13,29],
                   [14,30],
                   [15,31],
                   [16,32],
                   
                   #Level 2
                   [17,18],
                   [18,19],
                   [19,20],
                   [20,21],
                   [21,22],
                   [22,23],
                   [23,24],
                   [24,25],
                   [25,26],
                   [26,27],
                   [27,28],
                   [28,29],
                   [29,30],
                   [30,31],
                   [31,32],
                   [32,17],
                   
                   #Connecting L2 and 3
                   [17,33],
                   [18,34],
                   [19,35],
                   [20,36],
                   [21,37],
                   [22,38],
                   [23,39],
                   [24,40],
                   [25,41],
                   [26,42],
                   [27,43],
                   [28,44],
                   [29,45],
                   [30,46],
                   [31,47],
                   [32,48],
                   
                   #Level 3
                   [33,34],
                   [34,35],
                   [35,36],
                   [36,37],
                   [37,38],
                   [38,39],
                   [39,40],
                   [40,41],
                   [41,42],
                   [42,43],
                   [43,44],
                   [44,45],
                   [45,46],
                   [46,47],
                   [47,48],
                   [48,33],
                   [41,48],
                   
                   #Connecting L3 and 4
                   [41,57],
                   [40,56],
                   [39,55],
                   [38,54],
                   [37,53],
                   [36,52],
                   [35,51],
                   [34,50],
                   [33,49],
                   [48,58],
                   [34,49],
                   [34,51],
                   [36,51],
                   [36,53],
                   [38,53],
                   [38,55],
                   [40,55],
                   [40,57],
                   [48,49],
                   
                   
                   #Level 4
                   [49,50],
                   [50,51],
                   [51,52],
                   [52,53],
                   [53,54],
                   [54,55],
                   [55,56],
                   [56,57],
                   [57,58],
                   [58,49],
                   
                   #Connecting L4 and 5
                   [58,68],
                   [49,68],
                   [49,60],
                   [51,60],
                   [51,62],
                   [53,62],
                   [53,64],
                   [55,64],
                   [55,66],
                   [57,66],
                   [57,67],
                   [49,59],
                   [50,60],
                   [51,61],
                   [52,62],
                   [53,63],
                   [54,64],
                   [55,65],
                   [56,66],
                   
                   #Level 5
                   [59,60],
                   [60,61],
                   [61,62],
                   [62,63],
                   [63,64],
                   [64,65],
                   [65,66],
                   [66,67],
                   [68,59],
                   
                   #Connecting L5 and 6
                   [68,78],
                   [59,69],
                   [60,70],
                   [61,71],
                   [62,72],
                   [63,73],
                   [64,74],
                   [65,75],
                   [66,76],
                   [67,77],
                                      
                   #Level 6
                   [69,70],
                   [70,71],
                   [71,72],
                   [72,73],
                   [73,74],
                   [74,75],
                   [75,76],
                   [76,77],
                   [77,78],
                   [78,69],
                   
                   #Connecting L6 and 7
                   [78,94],
                   [69,79],
                   [70,80],
                   [71,81],
                   [72,82],
                   [73,83],
                   [74,84],
                   [75,85],
                   [76,86],
                   [77,87],
                                      
                   #Level 7
                   [79,80],
                   [80,81],
                   [81,82],
                   [82,83],
                   [83,84],
                   [84,85],
                   [85,86],
                   [86,87],
                   [87,88],
                   [88,89],
                   [89,90],
                   [90,91],
                   [91,92],
                   [92,93],
                   [93,94],
                   [94,79],
                   
                   #Connecting L7 and 8
                   [79,95],
                   [80,96],
                   [81,97],
                   [82,98],
                   [83,99],
                   [84,100],
                   [85,101],
                   [86,102],
                   [87,103],
                   [88,104],
                   [89,105],
                   [90,106],
                   [91,107],
                   [92,108],
                   [93,109],
                   [94,110],
                   
                   #Level 8
                   [95,96],
                   [96,97],
                   [97,98],
                   [98,99],
                   [99,100],
                   [100,101],
                   [101,102],
                   [102,103],
                   [103,104],
                   [104,105],
                   [105,106],
                   [106,107],
                   [107,108],
                   [108,109],
                   [109,110],
                   [110,95],
                   
                   #Landing Legs
                    [41,111],
                    [48,112],
                    [78,113],
                    [77,114]
                 ])

#Supports
restrainedDoF = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
                 17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
                 33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48] 
#Degrees of freedom restrained by suppports [i1, j1, k1, i2, j2, k2,...]

#Loading
forceVector = np.array([np.zeros(len(nodes)*3)]).T

g=9.81
mass = 358
accel = 6*g
forceTot = mass*accel*(-1)
forceMbr = forceTot/len(nodes)

#forceVector[i]=F   is how to index
#forceVector[3] = forceMbr
#forceVector[6] = forceMbr
fiter = 1
while fiter < len(nodes):
    forceVector[fiter*2] = forceTot
    fiter = fiter + 1

#print(forceVector)

#=====================End of Data Entry======================
#=====================Plotting Structure=====================
fig = plt.figure()
axes = fig.add_axes([.1,.1,3,3], projection='3d') #indicate a 3D plot
#fig.gca().set_aspect('equal', adjustable='box')
axes.view_init(20,35) #Setting viewing angle (rotation,elevation)


#Set offset distances for node labels
dx = .01
dy = .01
dz = .01

#Provide space/margin around structure
x_margin = .5
y_margin = .5
z_margin = .5

#Plotting members
for n, mbr in enumerate(members):
    node_i = mbr[0] #Node number for node i of this member
    node_j = mbr[1] #Node number for node j of this member
    
    ix = nodes[node_i-1,0] #x-coord of node i of this member
    iy = nodes[node_i-1,1] #y-coord of node i of this member
    iz = nodes[node_i-1,2] #z-coord of node i of this member
    jx = nodes[node_j-1,0] #x-coord of node j of this member
    jy = nodes[node_j-1,1] #y-coord of node j of this member
    jz = nodes[node_j-1,2] #z-coord of node j of this member
    label1 = n+1
    axes.plot3D([ix,jx],[iy,jy],[iz,jz],'black') #Member 3D member
    #axes.text((ix+jx)/2, (iy+jy)/2, (iz+jz)/2, label1, color='blue', fontsize=16) #Add node label
    


#Set axis limits to provide margin around structure
maxX = nodes.max(0)[0]
maxY = nodes.max(0)[1]
maxZ = nodes.max(0)[2]
minX = nodes.min(0)[0]
minY = nodes.min(0)[1]
minZ = nodes.min(0)[2]
axes.set_xlim([minX-x_margin, maxX+x_margin])
axes.set_ylim([minY-y_margin, maxY+y_margin])
axes.set_zlim([0, maxZ+z_margin])

#Plot nodes 
for n, node in enumerate(nodes):    #for loop with iterator
    axes.plot3D([node[0]],[node[1]],[node[2]],'bo',ms=6)
    label = str(n+1)
    axes.text(node[0]+dx,node[1]+dy,node[2]+dz, label, color='red', fontsize=16) #Add node label
    
axes.set_xlabel('X-Coord (m)',fontsize=25)
axes.set_ylabel('Y-Coord (m)',fontsize=25)
axes.set_zlabel('Z-Coord (m)',fontsize=25)
axes.set_title('Structure to Analyze',fontsize=25)
axes.grid()
plt.show()

#======================End of Inital Plotting=======================
#%%
#==============Calculating Member Orientation and Length==============
def memberOrientation3D(memberNo):    #Define function for calc
    memberIndex = memberNo-1  #Index identifying member in array of members
    node_i = members[memberIndex][0] #Node number for node i of this member
    node_j = members[memberIndex][1] #Node number for node j of this memeber
    
    ix = nodes[node_i-1][0] #x-coord for node i
    iy = nodes[node_i-1][1] #y-coord for node i
    iz = nodes[node_i-1][2] #z-coord for node i
    jx = nodes[node_j-1][0] #x-coord for node j
    jy = nodes[node_j-1][1] #y-coord for node j
    jz = nodes[node_j-1][2] #z-coord for node j
    
    #Angle of member wrt horizontal axis
    dx = jx-ix
    dy = jy-iy
    dz = jz-iz
    mag = math.sqrt((dx)**2 + (dy)**2 + (dz)**2)  #Magn of length of member


    cos_theta_x = (jx-ix)/mag
    cos_theta_y = (jy-iy)/mag
    cos_theta_z = (jz-iz)/mag
    return [cos_theta_x, cos_theta_y, cos_theta_z, mag]


#Calling function to calulate cosines and length for each member
cosX = np.array([]) #Initialise an array to hold cos(theta_x)
cosY = np.array([]) #Initialise an array to hold cos(theta_y)
cosZ = np.array([]) #Initialise an array to hold cos(theta_z)
lengths = np.array([])

for n, mbr in enumerate(members):
    [ctx, cty, ctz, length] = memberOrientation3D(n+1)
    cosX = np.append(cosX,ctx)
    cosY = np.append(cosY,cty)
    cosZ = np.append(cosZ,ctz)
    lengths = np.append(lengths, length)
#%%
#===================Calculating Stiffness Matrix=======================
def calculateKg3D(memberNo):
    
    x = cosX[memberNo-1]
    y = cosY[memberNo-1]
    z = cosZ[memberNo-1]
    mag = lengths[memberNo-1]

    #Defining individual elements of the lobal stiffness matrix
    #Row 1
    k11 = x**2
    k12 = x*y
    k13 = x*z
    k14 = -x**2
    k15 = -x*y
    k16 = -x*z
    #Row 2
    k21 = x*y
    k22 = y**2
    k23 = y*z
    k24 = -x*y
    k25 = -y**2
    k26 = -y*z
    #Row 3
    k31 = x*z
    k32 = y*z
    k33 = z**2
    k34 = -x*z
    k35 = -y*z
    k36 = -z**2
    #Row 4
    k41 = -x**2
    k42 = -x*y
    k43 = -x*z
    k44 = x**2
    k45 = x*y
    k46 = x*z
    #Row 5
    k51 = -x*y
    k52 = -y**2
    k53 = -y*z
    k54 = x*y
    k55 = y**2
    k56 = y*z
    #Row 6
    k61 = -x*z
    k62 = -y*z
    k63 = -z**2
    k64 = x*z
    k65 = y*z
    k66 = z**2
    
    #Build K11, K12, K21, K22
    K11 = (E*A/mag)*np.array([[k11,k12,k13],
                               [k21,k22,k23],
                               [k31,k32,k33]
                               ])
    K12 = (E*A/mag)*np.array([[k14,k15,k16],
                               [k24,k25,k26],
                               [k34,k35,k36]
                               ])
    K21 = (E*A/mag)*np.array([[k41,k42,k43],
                               [k51,k52,k53],
                               [k61,k62,k63]
                               ])
    K22 = (E*A/mag)*np.array([[k44,k45,k46],
                               [k54,k55,k56],
                               [k64,k65,k66]
                               ])
    return [K11, K12, K21, K22]

#Build Primary Stiffness Matrix Kp
nDoF = np.amax(members)*3 #Total number of DoFs
Kp = np.zeros([nDoF, nDoF]) #Initialize matrix

for n, mbr in enumerate(members):
    [K11, K12, K21, K22] = calculateKg3D(n+1) #Calling prev function
    node_i = mbr[0] #Node number for node i of this member
    node_j = mbr[1] #Node number for node i of this member
    
    #Primary matrix indices associated with each node
    ia = 3*node_i-3 #Index 1
    ib = 3*node_i-1 #Index 2
    ja = 3*node_j-3 #Index 3
    jb = 3*node_j-1 #Index 4
    
    Kp[ia:ib+1,ia:ib+1] = Kp[ia:ib+1,ia:ib+1] + K11
    Kp[ia:ib+1,ja:jb+1] = Kp[ia:ib+1,ja:jb+1] + K12
    Kp[ja:jb+1,ia:ib+1] = Kp[ja:jb+1,ia:ib+1] + K21
    Kp[ja:jb+1,ja:jb+1] = Kp[ja:jb+1,ja:jb+1] + K22


#Extract Structure Stiffness Matrix Ks
restrainedIndex = [x - 1 for x in restrainedDoF] #Index for each restrained DoF
#Delete rows and columns for restrained DoF
Ks = np.delete(Kp,restrainedIndex,0) #Delete rows
Ks = np.delete(Kp,restrainedIndex,1) #Delete columns
Ks = np.matrix(Ks) #Convert Ks from array to matrix
#=============Solving for Displacements and Reactions===================
forceVectorRed = copy.copy(forceVector) #Make copy of vetor to leave original
forceVetorRed = np.delete(forceVectorRed,restrainedIndex,0) #Delete rows corr. to restrained DoF
U = Ks.I*forceVectorRed

UG = np.zeros(nDoF) #Initialise array to hold displacement vector
c = 0 #Initialise counter to track how many restraints have been imposed
for i in np.arange(nDoF):
    if i in restrainedIndex:
        #Impose zero displacement
        UG[i] = 0
    else:
        #Assign actual displacement
        UG[i] = U[c]
        c=c+1
        
UG = np.array([UG]).T
FG = np.matmul(Kp,UG)

#Generate output statements
for i in np.arange(0,len(restrainedIndex)):
    index = restrainedIndex[i]

#Solving for member forces
mbrForces = np.array([])
for n, mbr in enumerate(members):
    x = cosX[n]
    y = cosY[n]
    z = cosZ[n]
    mag = lengths[n]
    
    node_i = mbr[0]
    node_j = mbr[1]
    
    ia = 3*node_i-3 #Index 1
    ib = 3*node_i-1 #Index 2
    ja = 3*node_j-3 #Index 3
    jb = 3*node_j-1 #Index 4
    
    #Trainsformation matrix
    T = np.array([[x,y,z,0,0,0],
                  [0,0,0,x,y,z]
                  ])
    
    disp = np.array([[UG[ia,0],  
                    UG[ia+1,0],  
                     UG[ib,0],  
                     UG[ja,0],  
                     UG[ja+1,0],  
                     UG[jb,0]  ]]).T
    disp_local = np.matmul(T,disp) #Local displacement
    F_axial = (A*E/mag)*(disp_local[1]-disp_local[0]) #Axial loads
    mbrForces = np.append(mbrForces,F_axial) #Store axial loads
#%%
#===========================Plotting Forces================================
    #Reminder to add yielded and buckling indicators
fig=plt.figure()
axes = fig.add_axes([.1,.1,2,2],projection='3d')
axes.view_init(20,35)

#Provide space/margin around structure
x_margin = .5
y_margin = .5
z_margin = .5

#Plotting members
for n, mbr in enumerate(members):
    node_i = mbr[0] #Node number for node i of this member
    node_j = mbr[1] #Node number for node j of this member
    
    ix = nodes[node_i-1,0] #x-coord of node i of this member
    iy = nodes[node_i-1,1] #y-coord of node i of this member
    iz = nodes[node_i-1,2] #z-coord of node i of this member
    jx = nodes[node_j-1,0] #x-coord of node j of this member
    jy = nodes[node_j-1,1] #y-coord of node j of this member
    jz = nodes[node_j-1,2] #z-coord of node j of this member
    
    axes.plot3D([ix,jx],[iy,jy],[iz,jz],'b') #Member 3D member

    #Index of DoF for this member
    ia = 3*node_i-3 #Index 1
    ib = 3*node_i-1 #Index 2
    ja = 3*node_j-3 #Index 3
    jb = 3*node_j-1 #Index 4
    
    if(abs(mbrForces[n])<.001):
        axes.plot3D([ix,jx],[iy,jy],[iz,jz],'grey',linestyle='--') #Zero force in member
    elif(mbrForces[n]>0):
        axes.plot3D([ix,jx],[iy,jy],[iz,jz],'b') #Member in tension
    else:
        axes.plot3D([ix,jx],[iy,jy],[iz,jz],'r') #Member in compression


#Set axis limits to provide margin around structure
maxX = nodes.max(0)[0]
maxY = nodes.max(0)[1]
maxZ = nodes.max(0)[2]
minX = nodes.min(0)[0]
minY = nodes.min(0)[1]
minZ = nodes.min(0)[2]
axes.set_xlim([minX-x_margin, maxX+x_margin])
axes.set_ylim([minY-y_margin, maxY+y_margin])
axes.set_zlim([0, maxZ+z_margin])
axes.set_xlabel('X-Coord (m)',fontsize=25)
axes.set_ylabel('Y-Coord (m)',fontsize=25)
axes.set_zlabel('Z-Coord (m)',fontsize=25)
axes.set_title('Tension/Compression Members',fontsize=25)
axes.grid()
plt.show()
    #%%
#================================Plotting Deflections==========================
fig=plt.figure()
axes = fig.add_axes([.1,.1,2,2],projection='3d')
axes.view_init(20,35)

#Provide space/margin around structure
x_margin = .5
y_margin = .5
z_margin = .5

#Plotting members
for mbr in members:
    node_i = mbr[0] #Node number for node i of this member
    node_j = mbr[1] #Node number for node j of this member
    
    ix = nodes[node_i-1,0] #x-coord of node i of this member
    iy = nodes[node_i-1,1] #y-coord of node i of this member
    iz = nodes[node_i-1,2] #z-coord of node i of this member
    jx = nodes[node_j-1,0] #x-coord of node j of this member
    jy = nodes[node_j-1,1] #y-coord of node j of this member
    jz = nodes[node_j-1,2] #z-coord of node j of this member
    

    #Index of DoF for this member
    ia = 3*node_i-3 #Index 1
    ib = 3*node_i-1 #Index 2
    ja = 3*node_j-3 #Index 3
    jb = 3*node_j-1 #Index 4
    
    axes.plot([ix,jx],[iy,jy],[iz,jz],'black', lw=.75) #Member
    axes.plot([ix + UG[ia,0]*xFac, jx + UG[ja,0]*xFac],
              [iy + UG[ib,0]*xFac, jy + UG[jb,0]*xFac],
              [iz + UG[ib,0]*xFac, jz + UG[jb,0]*xFac],'r')

#Set axis limits to provide margin around structure
maxX = nodes.max(0)[0]
maxY = nodes.max(0)[1]
maxZ = nodes.max(0)[2]
minX = nodes.min(0)[0]
minY = nodes.min(0)[1]
minZ = nodes.min(0)[2]
axes.set_xlim([minX-x_margin, maxX+x_margin])
axes.set_ylim([minY-y_margin, maxY+y_margin])
axes.set_zlim([0, maxZ+z_margin])
plt.legend(['Undeformed','Deflected with Factor of {one}'.format(one=xFac)],fontsize=20)
axes.set_xlabel('X-Coord (m)',fontsize=25)
axes.set_ylabel('Y-Coord (m)',fontsize=25)
axes.set_zlabel('Z-Coord (m)',fontsize=25)
axes.set_title('Deflected Shape',fontsize=25)
axes.grid()
plt.show()
#%%
#=========================Plotting Stresses===========================
fig = plt.figure()
axes = fig.add_axes([.1,.1,3,3], projection='3d') #indicate a 3D plot
#fig.gca().set_aspect('equal', adjustable='box')
axes.view_init(20,35) #Setting viewing angle (rotation,elevation)

#Set offset distances for node labels
dx = .01
dy = .01
dz = .01

#Provide space/margin around structure
x_margin = .5
y_margin = .5
z_margin = .5

#Plotting members
for n, mbr in enumerate(members):
    node_i = mbr[0] #Node number for node i of this member
    node_j = mbr[1] #Node number for node j of this member
    
    ix = nodes[node_i-1,0] #x-coord of node i of this member
    iy = nodes[node_i-1,1] #y-coord of node i of this member
    iz = nodes[node_i-1,2] #z-coord of node i of this member
    jx = nodes[node_j-1,0] #x-coord of node j of this member
    jy = nodes[node_j-1,1] #y-coord of node j of this member
    jz = nodes[node_j-1,2] #z-coord of node j of this member
    
    stress = round(mbrForces[n-1],0)
    axes.plot3D([ix,jx],[iy,jy],[iz,jz],'black') #Member 3D member
    axes.text((ix+jx)/2, (iy+jy)/2, (iz+jz)/2, stress, color='blue', fontsize=16) #Add node label
    
#Set axis limits to provide margin around structure
maxX = nodes.max(0)[0]
maxY = nodes.max(0)[1]
maxZ = nodes.max(0)[2]
minX = nodes.min(0)[0]
minY = nodes.min(0)[1]
minZ = nodes.min(0)[2]
axes.set_xlim([minX-x_margin, maxX+x_margin])
axes.set_ylim([minY-y_margin, maxY+y_margin])
axes.set_zlim([0, maxZ+z_margin])
axes.set_xlabel('X-Coord (m)',fontsize=25)
axes.set_ylabel('Y-Coord (m)',fontsize=25)
axes.set_zlabel('Z-Coord (m)',fontsize=25)
axes.set_title('Member Stresses',fontsize=25)
axes.grid()
plt.show()
    
    
#%%
#====================== Plotting Yielding and Buckling=====================
#Discovering maximum stresses
for n, mbr in enumerate(members):
    b_stress = (3.14159**2*E*r**2)/(4*lengths[n]**2)
    

fig = plt.figure()
axes = fig.add_axes([.1,.1,3,3], projection='3d') #indicate a 3D plot
#fig.gca().set_aspect('equal', adjustable='box')
axes.view_init(20,35) #Setting viewing angle (rotation,elevation)


#Set offset distances for node labels
dx = .01
dy = .01
dz = .01

#Provide space/margin around structure
x_margin = .5
y_margin = .5
z_margin = .5

for n, mbr in enumerate(members):
    node_i = mbr[0] #Node number for node i of this member
    node_j = mbr[1] #Node number for node j of this member
    
    ix = nodes[node_i-1,0] #x-coord of node i of this member
    iy = nodes[node_i-1,1] #y-coord of node i of this member
    iz = nodes[node_i-1,2] #z-coord of node i of this member
    jx = nodes[node_j-1,0] #x-coord of node j of this member
    jy = nodes[node_j-1,1] #y-coord of node j of this member
    jz = nodes[node_j-1,2] #z-coord of node j of this member
    stress = mbrForces[n-1]/A
    if abs(stress) > b_stress:
        axes.plot3D([ix,jx],[iy,jy],[iz,jz],'red') #Member 3D member
    elif abs(stress) > y_stress:
        axes.plot3D([ix,jx],[iy,jy],[iz,jz],'orange') #Member 3D member
    else:
        axes.plot3D([ix,jx],[iy,jy],[iz,jz],'black') #Member 3D member
        
#Set axis limits to provide margin around structure
maxX = nodes.max(0)[0]
maxY = nodes.max(0)[1]
maxZ = nodes.max(0)[2]
minX = nodes.min(0)[0]
minY = nodes.min(0)[1]
minZ = nodes.min(0)[2]
axes.set_xlim([minX-x_margin, maxX+x_margin])
axes.set_ylim([minY-y_margin, maxY+y_margin])
axes.set_zlim([0, maxZ+z_margin])
axes.set_xlabel('X-Coord (m)',fontsize=25)
axes.set_ylabel('Y-Coord (m)',fontsize=25)
axes.set_zlabel('Z-Coord (m)',fontsize=25)
axes.set_title('Yielding/Buckling Members',fontsize=25)
axes.grid()
plt.show()
#%%
#==========================Output Statements==============================
print("Reaction Forces:")
for i in np.arange(0,len(restrainedIndex)):
    index = restrainedIndex[i]
    print("Reaction at DoF {one}: {two} kN".format(one = index+1, two = round(FG[index].item()/1000,2)))
print("")
print("Member Forces")
for n, mbr in enumerate(members):
    print("Force in member {one} (nodes {two} to {three}) is {four} kN".format(one = n+1,
                                                                           two = mbr[0],
                                                                           three=mbr[1],
                                                                          four=round(mbrForces[n]/1000,2)))   
print("")
print("Nodal Displacements")
for n, node in enumerate(nodes):
    ix = 3*(n+1)-3 #x DoF for this node
    iy = 3*(n+1)-2 #y DoF for this node
    iz = 3*(n+1)-1 #z DoF for this node
    
    ux = round(UG[ix,0],5) #x nodal displacement
    uy = round(UG[iy,0],5) #y nodal displacement
    uz = round(UG[iz,0],5) #z nodal displacement
    print("Node {one}: Ux = {two} m, Uy = {three} m, Uz = {four} m".format(one=n+1,
                                                                           two=ux,
                                                                three=uy,
                                                                           four=ux))
print("")
print("Member Stresses")
for n, mbr in enumerate(members):
    print("Stress in member {one} (nodes {two} to {three}) is {four} kN".format(one = n+1,
                                                                           two = mbr[0],
                                                                 three=mbr[1],
                                                                          four=round((mbrForces[n]/A)/1000,2)))

print('')
print('Status of Members')
for n, mbr in enumerate(members):
    stress = mbrForces[n-1]/A
    if abs(stress) > b_stress:
        print('Status of Member {one}: Buckled'.format(one=n+1))
    elif abs(stress) > y_stress:
        print('Status of Member {one}: Yielded'.format(one=n+1))
    else:
        print('Status of Member {one}: Good'.format(one=n+1))




print(max(UG))
print(max((mbrForces/A)))
print(max(mbrForces))
