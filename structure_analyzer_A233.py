from pymatgen.core.structure import IStructure,Structure
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.defects.point_defects import ValenceIonicRadiusEvaluator#,VoronoiNN
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder,JMolCoordFinder,VoronoiAnalyzer,RelaxationAnalyzer,VoronoiConnectivity
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
##defining a function to judge if list a is in list b
def list_a_is_in_list_b(list_a,list_b):
    d1 = {}
    for i in list_a:
        if i in d1:
            d1[i]+=1
        else:
            d1[i]=1
    d2 = {}
    for i in list_b:
        if i in d2:
            d2[i]+=1
        else:
            d2[i]=1
    for key,value in d1.items():
        if key not in d2 or value > d2[key]:
            return False
    return True

s1 = Structure.from_file('POSCAR.orig')
s2 = Structure.from_file('CONTCAR.relax2')
spacegroup1 = SpacegroupAnalyzer(s1)
spacegroup2 = SpacegroupAnalyzer(s2)
print('space group symbol of structure1:', spacegroup1.get_space_group_symbol())
print('space group number of structure1:', spacegroup1.get_space_group_number())
print('space group symbol of structure2:', spacegroup2.get_space_group_symbol())
print('space group number of structure2:', spacegroup2.get_space_group_number())
Voronoi = VoronoiCoordFinder(s2,target=None)
site = s2.num_sites
#print("s2[0]:",s2.sites)
print("s2_cart_coords[0]",s2.cart_coords[0])
#print("s2_distance_(0,1)",s2.get_distance(0,1))
polyhedra = Voronoi.get_voronoi_polyhedra(1)
coordinate = Voronoi.get_coordination_number(1)
coordinate_sites = Voronoi.get_coordinated_sites(1)
Voronoi_Analyzer = VoronoiAnalyzer()
anay = Voronoi_Analyzer.analyze(s1,n=0)
strucs = [s1,s2]
anays = Voronoi_Analyzer.analyze_structures(strucs)
print("Voronoi_Analyzer.analyze(s1,n=0):",anay)
#plt = Voronoi_Analyzer.plot_vor_analysis(anays)
relax = RelaxationAnalyzer(s1,s2)
volume_change = relax.get_percentage_volume_change()
lattice_parameter_changes = relax.get_percentage_lattice_parameter_changes()
print('initial volume:',s1.volume)
print('final volume:',s2.volume)
print("percent_volume_change:",volume_change)
print("percent_lattice_change:",lattice_parameter_changes)
bond_dist = relax.get_percentage_bond_dist_changes(max_radius=6)
print("percent_bond_distance_change:",bond_dist)
connec = VoronoiConnectivity(s2)
#print("connec.get_connections():",connec.get_connections())


#A_233_dopant = [0]
A_233_dopant_neighbour = [[164,173],[163,172],[152,155],[151,154],[154,155],[172,173],[151,152],[163,164]]
A_233_dopant_neighbour_MA_I_a = [[164,173],[163,172],[143,146],[142,145],[145,146],[163,164],[142,143],[172,173]]
A_233_dopant_neighbour_MA_I_b = [[182,191],[176,185],[176,182],[185,191],[166,167],[148,149],[157,158],[154,155]]
A_233_dopant_neighbour_MA_I_c = [[180,189],[177,186],[186,189],[177,180],[171,173],[150,152],[162,164],[153,155]]

###Calculate dopant bond length
dopant_bond_a = 0
for i in range(2):
    #print("s2.cart_coords[A_233_dopant_neighbour[i][0]]:",list(s2.cart_coords[A_233_dopant_neighbour[i][0]]))
    a1=np.sqrt(sum((list(s2.cart_coords[A_233_dopant_neighbour[i][0]])[j]-list(s2.cart_coords[A_233_dopant_neighbour[i][-1]])[j])**2 for j in range(len(list(s2.cart_coords[A_233_dopant_neighbour[i][0]])))))
    #print("a1:",a1)
    dopant_bond_a+=a1
print("dopand_bond_a:",dopant_bond_a/2)

dopant_bond_b = 0
for i in range(2,4):
    b1=np.sqrt(sum((list(s2.cart_coords[A_233_dopant_neighbour[i][0]])[j]-list(s2.cart_coords[A_233_dopant_neighbour[i][-1]])[j])**2 for j in range(len(list(s2.cart_coords[A_233_dopant_neighbour[i][0]])))))
    #print("s2.cart_coords[A_233_dopant_neighbour[{}][0]:".format(i),list(s2.cart_coords[A_233_dopant_neighbour[i][0]]))
    #print("b1:",b1)
    dopant_bond_b+=b1
print("dopand_bond_b:",dopant_bond_b/2)

dopant_bond_c = 0
for i in range(4,8):
    c1=np.sqrt(sum((list(s2.cart_coords[A_233_dopant_neighbour[i][0]])[j]-list(s2.cart_coords[A_233_dopant_neighbour[i][-1]])[j])**2 for j in range(len(list(s2.cart_coords[A_233_dopant_neighbour[i][0]])))))
    #print("s2.cart_coords[A_233_dopant_neighbour[i]:",list(s2.cart_coords[A_233_dopant_neighbour[i][-1]]))
    #print("c1:",c1)
    dopant_bond_c+=c1
print("dopand_bond_c:",dopant_bond_c/4)

### To calculate the A direction octahedra a,b,c length
dopant_bond_A_a = 0
for i in range(2):
    #print("s2.cart_coords[A_233_dopant_neighbour[i][0]]:",list(s2.cart_coords[A_233_dopant_neighbour[i][0]]))
    a2=np.sqrt(sum((list(s2.cart_coords[A_233_dopant_neighbour_MA_I_a[i][0]])[j]-list(s2.cart_coords[A_233_dopant_neighbour_MA_I_a[i][-1]])[j])**2 for j in range(len(list(s2.cart_coords[A_233_dopant_neighbour_MA_I_a[i][0]])))))
    #print("a1:",a1)
    dopant_bond_A_a+=a2
print("dopand_bond_A_a:",dopant_bond_A_a/2)

dopant_bond_A_b = 0
for i in range(2,4):
    b2=np.sqrt(sum((list(s2.cart_coords[A_233_dopant_neighbour_MA_I_a[i][0]])[j]-list(s2.cart_coords[A_233_dopant_neighbour_MA_I_a[i][-1]])[j])**2 for j in range(len(list(s2.cart_coords[A_233_dopant_neighbour_MA_I_a[i][0]])))))
    #print("a1:",a1)
    dopant_bond_A_b+=b2
print("dopand_bond_A_b:",dopant_bond_A_b/2)

dopant_bond_A_c = 0
for i in range(4,8):
    c2=np.sqrt(sum((list(s2.cart_coords[A_233_dopant_neighbour_MA_I_a[i][0]])[j]-list(s2.cart_coords[A_233_dopant_neighbour_MA_I_a[i][-1]])[j])**2 for j in range(len(list(s2.cart_coords[A_233_dopant_neighbour_MA_I_a[i][0]])))))
    #print("a1:",a1)
    dopant_bond_A_c+=c2
print("dopand_bond_A_c:",dopant_bond_A_c/4)

### To calculate B direction a.b,c
dopant_bond_B_a = 0
for i in range(2):
    a3=np.sqrt(sum((list(s2.cart_coords[A_233_dopant_neighbour_MA_I_b[i][0]])[j]-list(s2.cart_coords[A_233_dopant_neighbour_MA_I_b[i][-1]])[j])**2 for j in range(len(list(s2.cart_coords[A_233_dopant_neighbour_MA_I_b[i][0]])))))
    #print("a1:",a1)
    dopant_bond_B_a+=a3
print("dopand_bond_B_a:",dopant_bond_B_a/2)

dopant_bond_B_b = 0
for i in range(2,4):
    b3=np.sqrt(sum((list(s2.cart_coords[A_233_dopant_neighbour_MA_I_b[i][0]])[j]-list(s2.cart_coords[A_233_dopant_neighbour_MA_I_b[i][-1]])[j])**2 for j in range(len(list(s2.cart_coords[A_233_dopant_neighbour_MA_I_b[i][0]])))))
    #print("a1:",a1)
    dopant_bond_B_b+=b3
print("dopand_bond_B_b:",dopant_bond_B_b/4)

dopant_bond_B_c = 0
for i in range(4,8):
    c3=np.sqrt(sum((list(s2.cart_coords[A_233_dopant_neighbour_MA_I_b[i][0]])[j]-list(s2.cart_coords[A_233_dopant_neighbour_MA_I_b[i][-1]])[j])**2 for j in range(len(list(s2.cart_coords[A_233_dopant_neighbour_MA_I_b[i][0]])))))
    #print("a1:",a1)
    dopant_bond_B_c+=c3
print("dopand_bond_B_c:",dopant_bond_B_c/4)

###To calculate the C direction a,b,c

dopant_bond_C_a = 0
for i in range(2):
    a4=np.sqrt(sum((list(s2.cart_coords[A_233_dopant_neighbour_MA_I_b[i][0]])[j]-list(s2.cart_coords[A_233_dopant_neighbour_MA_I_b[i][-1]])[j])**2 for j in range(len(list(s2.cart_coords[A_233_dopant_neighbour_MA_I_b[i][0]])))))
    #print("a1:",a1)
    dopant_bond_C_a+=a4
print("dopand_bond_C_a:",dopant_bond_C_a/2)

dopant_bond_C_b = 0
for i in range(2,4):
    b4=np.sqrt(sum((list(s2.cart_coords[A_233_dopant_neighbour_MA_I_b[i][0]])[j]-list(s2.cart_coords[A_233_dopant_neighbour_MA_I_b[i][-1]])[j])**2 for j in range(len(list(s2.cart_coords[A_233_dopant_neighbour_MA_I_b[i][0]])))))
    #print("a1:",a1)
    dopant_bond_C_b+=b4
print("dopand_bond_C_b:",dopant_bond_C_b/4)

dopant_bond_C_c = 0
for i in range(4,8):
    c4=np.sqrt(sum((list(s2.cart_coords[A_233_dopant_neighbour_MA_I_b[i][0]])[j]-list(s2.cart_coords[A_233_dopant_neighbour_MA_I_b[i][-1]])[j])**2 for j in range(len(list(s2.cart_coords[A_233_dopant_neighbour_MA_I_b[i][0]])))))
    #print("a1:",a1)
    dopant_bond_C_c+=c4
print("dopand_bond_C_c:",dopant_bond_C_c/4)

average_MA_a = (np.sum([dopant_bond_A_a/2,dopant_bond_B_a/2,dopant_bond_C_a/2]))/3
#print("average_MA_a:",average_MA_a)
average_MA_b = (np.sum([dopant_bond_A_b/2,dopant_bond_B_b/4,dopant_bond_C_b/4]))/3
average_MA_c = (np.sum([dopant_bond_A_c/4,dopant_bond_B_c/4,dopant_bond_C_c/4]))/3

diff_dopant_neig_a = dopant_bond_a/2-average_MA_a
diff_dopant_neig_b = dopant_bond_b/2-average_MA_b
diff_dopant_neig_c = dopant_bond_c/4-average_MA_c


print("diff_dopant_neig_a:",diff_dopant_neig_a)
print("diff_dopant_neig_b:",diff_dopant_neig_b)
print("diff_dopant_neig_c:",diff_dopant_neig_c) 


