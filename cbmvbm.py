import numpy as np
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.inputs import Kpoints,Poscar,Incar,Potcar #Kpoints_support_modes
from pymatgen.core.structure import IStructure
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine,Kpoint
vasp = Vasprun('vasprun.xml')#,ionic_step_skip=1,ionic_step_offset=1,parse_dos=True,parse_eigen=True,occu_tol=1e-8)
efermi = vasp.efermi
eigenvals = vasp.eigenvalues
bs = vasp.get_band_structure(kpoints_filename='KPOINTS',efermi=efermi,line_mode=True)
#kpoints_modes = Kpoints_support_modes(3) #Line mode
kpoints = Kpoints.from_file('KPOINTS')
poscar = Poscar.from_file('POSCAR')
incar = Incar.from_file(('INCAR'))
struc = IStructure.from_file('POSCAR')
lattice = struc.lattice
labels = kpoints.labels
space_group = struc.get_space_group_info()
#coords = struc.frac_coords()
BS = BandStructure(kpoints.kpts,eigenvals,lattice,efermi,structure = struc)#kpoints.kpts
labels_dict = BS.labels_dict
BSSL = BandStructureSymmLine(kpoints.kpts,eigenvals,lattice,efermi,labels_dict,structure = struc)
b = vasp.eigenvalue_band_properties
vbm = BS.get_vbm()
cbm = BS.get_cbm()
print(b,efermi,'cbm:',bs.get_cbm(),'vbm:',bs.get_vbm())#,cbm,vbm#BSSL.get_branch(90)#,kpoints.kpts
#print a.eigenvalue_band_properties,fermi
#vbm = a.eigenvalue_band_properties
