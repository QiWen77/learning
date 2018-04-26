from pymatgen.core.structure import Structure
from pymatgen.io.vasp.sets import MPStaticSet,MPRelaxSet,MPNonSCFSet,MITMDSet

struc = Structure.from_file('MAI-2FA.cif')
struct = struc.to(fmt="poscar",filename='POSCAR')
#pri_struct = struct.get_primitive_structure()
#print(pri_struct)
#vis = MPRelaxSet(struct)
#vis=MPStaticSet(struct)
#vis = MPNonSCFSet(struct)
vis = MITMDSet(struc,300,300,100000)#(self, structure, start_temp, end_temp, nsteps, time_step=2,spin_polarized=False, **kwargs)
vis.write_input('.')

