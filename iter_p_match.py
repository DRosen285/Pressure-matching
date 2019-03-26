import numpy as np
import scipy
from numpy import*
np.set_printoptions(threshold='nan')
from itertools import izip
from scipy import linalg

# script to compute self consistent pressure matching correction (after p_match.py has been performed)

p_icg=loadtxt("icg_pressures_280K.xvg",unpack=True,usecols=[0])#virial per frame of the cg run with the DA barostat; note: virial does not contain additional f_v term (non corrected virial)
vol_cg=loadtxt("icg_volumes_280K.xvg",unpack=True,usecols=[0]) #volume per frame of the cg run with th DA barostat
vol_atom=loadtxt("aa_volumes.xvg",unpack=True,usecols=[1]) #volume per frame of the atomstic system
p_ref=loadtxt("aa_pressures.xvg",unpack=True,usecols=[1]) #volume per frame of the atomstic system
N=#number of CG beads
n=#number of FG atoms
n_frame_aa = len(p_ref)
n_frame_cg= len(p_icg)
f_vol1=np.zeros(shape=(2,2))#2x2 matrix containing linear independent equations to solve the equations
f_vol1_2=np.zeros(shape=(2,2))#2x2 matrix containing linear independent equations to solve the equations
p_diff=np.zeros(2) #2d-vector containing the pressure difference between the atomistic and the coarse grained system
p_diff_2=np.zeros(2) #2d-vector containing the pressure difference between the atomistic and the coarse grained system
basis=np.zeros(2) #2d-vector containing the basis functions for volume dependent part of the potential
basis_2=np.zeros(2) #2d-vector containing the basis functions for volume dependent part of the potential
#convert bar into kJ/mol/nm^3
prefactor=1/16.6054

# calculate average volume of atomistic trajcetory
v_sum=0
for i in range(0,len(vol_atom)):
 v_sum+=vol_atom[i]
vol_atom_avg=v_sum/len(vol_atom)

#calculate average volume of cg trajcetory
v_sum=0
for i in range(0,len(vol_cg)):
 v_sum+=vol_cg[i]
vol_cg_avg=v_sum/len(vol_cg)

# solve linear least squares problem of type A*x=b

# determine coefficients for the aa system (aa pV -diagram)
basis[0]=-N/vol_atom_avg
for i in range(0,n_frame_aa):
 basis[1]=-2*N*(vol_atom[i]-vol_atom_avg)/(vol_atom_avg*vol_atom_avg)
 for d in range(0,2):
  p_diff[d]+=(prefactor*p_ref[i])*basis[d] #populate vector b
  for dp in range(0,2):
   f_vol1[d,dp]+=basis[d]*basis[dp] #populate matrix A
# averge over number of frames
b=p_diff/n_frame_aa
A=f_vol1/n_frame_aa
# coefficients in kj/mol
c=np.linalg.solve(A,b) #solve equation for vector x(=c)
print (c)
# coefficients in bar
print(c/prefactor)
# coefficients in bar based on volume in Angstrom^3 (as used in lammps)
print (c*1000/prefactor)

# solve linear equations for cg_trajectory

# determine coefficients for the cg system (cg run with DA barostat) (aa cg -diagram)
basis_2[0]=-N/vol_atom_avg
for i in range(0,n_frame_cg):
 basis_2[1]=-2*N*(vol_cg[i]-vol_atom_avg)/(vol_atom_avg*vol_atom_avg)
 for d_2 in range(0,2):
  p_diff_2[d_2]+=(prefactor*p_icg[i])*basis_2[d_2] #populate vector b
  for dp_2 in range(0,2):
   f_vol1_2[d_2,dp_2]+=basis_2[d_2]*basis_2[dp_2] #populate matrix A
# averge over number of frames
b_2=p_diff_2/n_frame_cg
A_2=f_vol1_2/n_frame_cg
# coefficients in kj/mol
c_2=np.linalg.solve(A_2,b_2) #solve equation for vector x(=c)
# coefficients in kj/mol
print (c_2)
# coefficients in bar
print(c_2/prefactor)
#coefficients in bar based on volume in Angstrom^3 (as used in lammps)
#print (c_2*1000/prefactor)
print ((c/prefactor)-c_2/prefactor)

#if virial (p_icg) contains additional volume dependent Force term, then:
# c=(c-c_2)+c(DA (obtained from p_match.py))

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
# A_11 A_12    x_1     b_1 (where A_11 = A_1*A_1)
#           x       =
# A_21 A_22    x_2     b_2

# the two linear equations to be solved:
# 1) A_11 *x_1 + A_12*x_2 = b_1
# 2) A_21 *x_1 + A_22*x_2 = b_2

####################################################
