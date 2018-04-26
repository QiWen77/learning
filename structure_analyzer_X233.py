from pymatgen.core.structure import IStructure,Structure
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.defects.point_defects import ValenceIonicRadiusEvaluator#,VoronoiNN
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder,JMolCoordFinder,VoronoiAnalyzer,RelaxationAnalyzer,VoronoiConnectivity
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

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
print("s2[197]:",s2[197])
polyhedra = Voronoi.get_voronoi_polyhedra(1)
coordinate = Voronoi.get_coordination_number(1)
coordinate_sites = Voronoi.get_coordinated_sites(1)
Voronoi_Analyzer = VoronoiAnalyzer()
anay = Voronoi_Analyzer.analyze(s2,n=0)
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
bond_dist = relax.get_percentage_bond_dist_changes()
connec = VoronoiConnectivity(s2)


B_222_dopant = [197]
B_222_dopant_neighbour = [110,116]
B_222_dopant_neighbour_Pb = [172,166,163]
B_222_dopant_neighbour_Pb_I = [[119,125],[113,110],[115,109]]
B_222 = zip(B_222_dopant_neighbour_Pb,B_222_dopant_neighbour_Pb_I)
#print("B_222:",B_222.__next__())
### To determine the dopant and its neighbour I pair and bond distance
comb3 = [ ]
for i in B_222_dopant:
    for j in B_222_dopant_neighbour:
        pair = [ ]
        pair.append(i)
        pair.append(j)
        #print("pair:",pair)
        comb3.append(pair)
print("comb3 X_222 dopant and neighbours combination:",comb3)

neighbour3 = [ ]
for i in comb3:
    for j in connec.get_connections():
        if list_a_is_in_list_b(i,j):
            print("j:",j)
            neighbour3.append(j)
print("neighbour3:",neighbour3)
pop_list = range(1,len(comb3)+1)
for i in pop_list:
    neighbour3.pop(i)
print("neighbour3 B_222 dopant and neighbours and distance:",neighbour3)
dopant_bond_a = neighbour3[0][-1]+neighbour3[1][-1]
#dopant_bond_c = neighbour3[2][-1]
average_dopant_bond_a = dopant_bond_a/2
#average_dopant_bond_b = dopant_bond_b/2
#average_dopant_bond_c = dopant_bond_c/2
distance3 = []
for i in neighbour3:
    distance3.append(i[-1])
average_dopant_bond = sum(distance3)/len(distance3)
print("\n","B_222 average_dopant_bond_a:",average_dopant_bond_a)#,"\n","B_222 average_dopant_bond_b:",average_dopant_bond_b)#,"\n","B_222 average_dopant_bond_c:",average_dopant_bond_c,"\n","B_222 average_dopant_bond:",average_dopant_bond)

###To determine the adjacent Pb and its I pair and bond distance
B_222_comb = [ ]
for i in B_222:
    B_222_comb.append(tuple(i))
print("B_222 adjacent combinations:",B_222_comb)

comb4 = [ ]
for i in B_222_comb:
   # comb2 = [ ]
    center_Pb = i[0]
    coordination_I = i[1]
    #print("center_Pb:",center_Pb,"\n","coordination_I:",coordination_I)
    for j in coordination_I:
        temp = []
        temp.append(center_Pb)
        temp.append(j)
       # print()
        comb4.append(temp)
print("B_222 adjacent Pb and neighbour combination:",comb4)
#print("connec.get_connections():",connec.get_connections())

neighbour4 = []
for i in comb4:
    for j in connec.get_connections():
        if list_a_is_in_list_b(i,j):
           # print(j)
            neighbour4.append(j)
print("(neighbour4):",neighbour4)
# To eliminate the redundant pairs
pop_list = range(1,len(comb4)+1)
#print("pop_list:",pop_list)

for i in pop_list:
    neighbour4.pop(i)
print("B_222 adjacent Pb and neighbour combination and distances:",neighbour4)

distance_adjacent_a = [ ]
for i in neighbour4:
    distance_adjacent_a.append(i[-1])
average_distance_adjacent_a = sum(distance_adjacent_a)/len(distance_adjacent_a)
print("B_222 average_distance_adjacent_a:",average_distance_adjacent_a)

print("B_222 average_dopant_bond_a-average_distance_adjacent_a:",average_dopant_bond_a-average_distance_adjacent_a)














# max_bond = connec.get_max_bond_lengths(s2)
# print volume_change,lattice_parameter_changes,bond_dist
#valence =  BVAnalyzer().get_valences(s2)
#oxide_state = BVAnalyzer().get_oxi_state_decorated_structure(s2)
# print "percent volume change(0.055 implies a 5.5% increase):",volume_change
# print "percent lattice change(-0.055 implies a 5.5% decrease):",lattice_parameter_changes
# print "percent bond distance change(0.055 implies a 5.5% increase):",bond_dist[1]
# print "valences of structure:",valence
# print "oxidation states:",oxide_state
#VIRE = ValenceIonicRadiusEvaluator(s2)
#radii = VIRE.radii
#get_radii = VIRE._get_ionic_radii()
#ox_structure = VIRE._get_valences()
# print ox_structure
# target = ["I","Pb","K","C","H","N"]
#NN = VoronoiNN(targets=None)
#cn = NN.get_nn_info(s2,0)
#poly = NN.get_voronoi_polyhedra(s2,0)
# nn = NN.get_nn(s2,0)
# wnn = NN.get_weights_of_nn_sites(s2,0)
# img = NN.get_nn_images(s2,0)
#nn_info = NN.get_nn_info(s2,0)
#print(poly)#,nn_info

