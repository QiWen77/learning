from Element_plus import list_filter_elements
from pymatgen.core.structure import Structure
from atomate.vasp.workflows.presets.core import wf_bandstructure,wf_bandstructure_plus_hse,wf_nudged_elastic_band
from fireworks import LaunchPad 
from atomate.vasp.powerups import add_modify_incar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

#structure1 = Structure.from_file("POSCAR_initial")
#structure2 = Structure.from_file("POSCAR_final")
#structures = [structure1,structure2]
structure = Structure.from_file("POSCAR")
structure.make_supercell([2,3,1])
els_p1 = list_filter_elements(lambda el:1 in el.common_oxidation_states)
els_p2 = list_filter_elements(lambda el:2 in el.common_oxidation_states)
els_n1 = list_filter_elements(lambda el:-1 in el.common_oxidation_states)
els_p1_pop = [0,2,3,5,8,8]
for i in els_p1_pop:
    els_p1.pop(i)
append_list = ['Au','Cu']
els_p1=els_p1+append_list
els_p2_pop = [2,10,-1,-1,-1]
for i in els_p2_pop:
    els_p2.pop(i)
els_n1_pop = [0,0,-1,-1]
for i in els_n1_pop:
    els_n1.pop(i)
FA = [76,124,130,10,16,4,22,28]
structure1 = structure
structure1.remove_sites(FA)
site_C5 = [0.66667,0.55556,0.15493]
for i in list(range(len(els_p1))):
    structure1.append(els_p1[i],site_C5)
    structure1.to(fmt="poscar",filename="POSCAR_A_{}".format(els_p1[i]))
    
    wf = wf_bandstructure(structure1)
    lpad = LaunchPad.auto_load()
    lpad.add_wf(wf)
    structure1.remove_sites([136])
structure = Structure.from_file("POSCAR")
structure.make_supercell([2,3,1])
Pb = [70]
structure2=structure
structure2.remove_sites(Pb)
site_Pb11 = [0.5,0.33333,0.506]
for i in list(range(len(els_p2))):
    structure2.append(els_p2[i],site_Pb11)
    structure2.to(fmt="poscar",filename="POSCAR_B_{}".format(els_p2[i]))
    wf = wf_bandstructure(structure2)
    lpad = LaunchPad.auto_load()
    lpad.add_wf(wf)
    structure2.remove_sites([143])
structure = Structure.from_file("POSCAR")
structure.make_supercell([2,3,1])
I30 = [113]
I32 = [115]
site_I30 = [0.58442,0.77922,0.751]
site_I32 = [0.33117,0.61039,0.751]
structure3 = structure
structure3.remove_sites(I30)
for i in list(range(len(els_n1))):
    structure3.append(els_n1[i],site_I30) 
    structure3.to(fmt="poscar",filename="POSCAR_I30_{}".format(els_n1[i]))
    wf = wf_bandstructure(structure3)
    lpad = LaunchPad.auto_load()
    lpad.add_wf(wf)
    structure3.remove_sites([143])
structure = Structure.from_file("POSCAR")
structure.make_supercell([2,3,1])
structure4 = structure
structure4.remove_sites(I32)
for i in list(range(len(els_n1))):
    structure4.append(els_n1[i],site_I32)
    structure4.to(fmt="poscar",filename="POSCAR_I32_{}".format(els_n1[i]))
    wf = wf_bandstructure(structure4)
    lpad = LaunchPad.auto_load()
    lpad.add_wf(wf)
    structure4.remove_sites([143])
print(els_p1,els_p2,els_n1)
#s=structure
#spacegroup = SpacegroupAnalyzer(s)
#print('space group symbol:', spacegroup.get_space_group_symbol())
#print('space group number:', spacegroup.get_space_group_number())
#wf = wf_bandstructure_plus_hse(structure,gap_only=False)
#wf = wf_bandstructure(s)
#wf = wf_nudged_elastic_band(structures,parent,user_incar_settings = [{"NSW":150}])
#wf = add_modify_incar(wf, {'incar_update': {'SIGMA': 0.2, 'ISMEAR': 1}}, fw_name_constraint='optimization')
#lpad = LaunchPad.auto_load()
#lpad.add_wf(wf)
