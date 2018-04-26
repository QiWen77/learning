from pymatgen.core import Structure
import os
#import shutil
#from Element_plus import list_filter_elements
from mpworks.submission.submission_mongo import SubmissionMongoAdapter
from pymatgen import MPRester
from pymatgen.matproj.snl import StructureNL
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
structure = Structure.from_file("POSCAR")
s = structure
#structure.make_supercell([1,3,3])
#structure.to(fmt="poscar",filename="POSCAR")
#els_p3 = list_filter_elements(lambda el:3 in el.common_oxidation_states)
#els_n2 = list_filter_elements(lambda el:-2 in el.common_oxidation_states)
#els_n2.pop(0)
#poplist=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,14,14,14,14,14,14]
#for p in poplist:
#    els_p3.pop(p)
#for B in els_p3:
#    structure[74]=B
#    for X in els_n2:
#        structure[96]=X
#els_n1.append("Cu")
#els_n1.append("Au")
#print els_n1
#MA =[17,35,53,71,89,107,143,215] #2*3*3 system
#MA = [7,15,23,31,39,47,63,95]
#FA = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,18,19,20,30,31,32,33,34,35]#alpha
#FA = [0,1,2,3,4,5,6,7,8,9,12,13,20,21,22,23]#delta
#FA = [0,1,2,3,4,5,6,7]
#s.remove_sites(FA)
#add = [0.7469,0.74726,0.7469] # for 2*2*2 system
#add  = [0.74626,0.830865,0.8315]
#add1 = [0.66667,0.33333,0.34262]#alpha
#add1 = [0.33333,0.66667,0.15493]#delta
#add = [0.5,0.56901,0.5]
#s.append('K',add)#els_n1[9
#add2 = [0.33333,0.66667,0.61381]#alpha
#add2 = [0.66667,0.33333,0.65493]#delta
#s.append('K',add2)
#add3 = [0,0,0.010678]#alpha
#s.append('K',add3)
#s.to(fmt="poscar",filename="POSCAR4")
spacegroup = SpacegroupAnalyzer(s)
print('space group symbol:', spacegroup.get_space_group_symbol())
print('space group number:', spacegroup.get_space_group_number())
sma = SubmissionMongoAdapter.auto_load()
snl = StructureNL(s, authors='WenQi <qiwen930110@gmail.com>',remarks=['MAPbI3', 'B-site'])
parameters = {'boltztrap':False,'force_gamma':True}
sma.submit_snl(snl, 'qiwen930110@gmail.com', parameters=parameters)
