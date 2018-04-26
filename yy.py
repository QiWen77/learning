import pymatgen as mg
import os
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.sets import VaspInputSet,DictSet,MPRelaxSet,MPStaticSet,MPSOCSet
import json
import os
pos=Poscar.from_file('./POSCAR')
struc=pos.structure
MPRelax=MPRelaxSet(struc)
#inputs=MPRelax.write_input('.')
MPSta=MPStaticSet(MPRelax)
