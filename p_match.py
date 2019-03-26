import numpy as np
import scipy
from numpy import*
np.set_printoptions(threshold='nan')
from itertools import izip
from scipy import linalg

# script to perform pressure matching

p_ref=loadtxt("aa_pressures.xvg",unpack=True,usecols=[1]) #virial per frame of the atomistic reference run
p_cg=loadtxt("cg_pressures.xvg",unpack=True,usecols=[1]) #virial per frame of the rerun of the atomistic trajectory with the cg potential
vol_atom=loadtxt("aa_volumes.xvg",unpack=True,usecols=[1]) #volume per frame of the atomstic system
N=#number of beads
n_frame = len(p_cg)
f_vol1=np.zeros(shape=(2,2))#2x2 matrix containing linear independent equations to solve the equations
p_diff=np.zeros(2) #2d-vector containing the pressure difference between the atomistic and the coarse grained system
basis=np.zeros(2) #2d-vector containing the basis functions for volume dependent part of the potential
#convert bar into kJ/mol/nm^3
prefactor=1/16.6054

# calculate average volume of atomistic trajcetory
v_sum=0
for i in range(0,len(vol_atom)):
 v_sum+=vol_atom[i]
vol_atom_avg=v_sum/len(vol_atom)

# solve linear least squares problem of type A*x=b

basis[0]=-N/vol_atom_avg
for i in range(0,n_frame):
 basis[1]=-2*N*(vol_atom[i]-vol_atom_avg)/(vol_atom_avg*vol_atom_avg)
 for d in range(0,2):
  p_diff[d]+=(prefactor*p_ref[i]-prefactor*p_cg[i])*basis[d] #populate vector b
  for dp in range(0,2):
   f_vol1[d,dp]+=basis[d]*basis[dp] #populate matrix A
#averge over number of frames
b=p_diff/n_frame
A=f_vol1/n_frame
# coefficients in kj/mol
c=np.linalg.solve(A,b) #solve equation for vector x(=c)
print (c)
# coefficients in bar
print(c/prefactor)
# coefficients in bar based on volume in Angstrom^3 (as used in lammps)
print (c*1000/prefactor)


####### theoretical framework of the code #########

# based on Das,Andersen J. Chem. Phys. 132 (2010) and Dunn,Noid J. Chem. Phys. 143 (2015)
# with the help of Joe Rudzinski
# more advanced version available within the BOCS simulation package:
# https://github.com/noid-group/BOCS
# https://pubs.acs.org/doi/10.1021/acs.jpcb.7b09993

# linear problem A*x = b -> solve for x
# where A is a matrix containing m-equations and x a vector containing n coefficients
# reformulate problem as a sum:
# sum (from j=1 to n) A_ij*x_j=b_i (i=1,....m)
# where A = 2x2 Matrix-> resulting matrix equations:
# A_11 A_12    x_1     b_1
#           x       =
# A_21 A_22    x_2     b_2

# the two linear equations to be solved:
# 1) A_11 *x_1 + A_12*x_2 = b_1
# 2) A_21 *x_1 + A_22*x_2 = b_2

####################################################
