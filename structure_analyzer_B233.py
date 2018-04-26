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
s2 = Structure.from_file('CONTCAR')
spacegroup1 = SpacegroupAnalyzer(s1)
spacegroup2 = SpacegroupAnalyzer(s2)
print('space group symbol of structure1:', spacegroup1.get_space_group_symbol())
print('space group number of structure1:', spacegroup1.get_space_group_number())
print('space group symbol of structure2:', spacegroup2.get_space_group_symbol())
print('space group number of structure2:', spacegroup2.get_space_group_number())
Voronoi = VoronoiCoordFinder(s2,target=None)
site = s2.num_sites
print("s2[0]:",s2[0])
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
bond_dist = relax.get_percentage_bond_dist_changes()
connec = VoronoiConnectivity(s2)

B_233_dopant = [0]
B_233_dopant_neighbour = [144,153,162,165,180,181]
B_233_dopant_neighbour_Pb = [117,111,114,109,110]
B_233_dopant_neighbour_Pb_I =[[144,153,171,174,189,190],[147,156,165,168,183,184],[150,159,162,168,186,187],[145,154,163,166,181,182],[146,155,164,167,180,182]]
B_233 = zip(B_233_dopant_neighbour_Pb,B_233_dopant_neighbour_Pb_I)


### To determine the dopant and its neighbour I pair and bond distance 
comb1 = []
for i in B_233_dopant:
    for j in B_233_dopant_neighbour:
        pair = []
        pair.append(i)
        pair.append(j)
        #print("pair:",pair)
        comb1.append(pair)
#print("B_233 dopant and neighbours combination:",comb1)

#print("get_connections:",connec.get_connections())
neighbour1 = [ ]
for i in comb1:
    for j in connec.get_connections():
        if list_a_is_in_list_b(i,j):
            #print(j)
            neighbour1.append(j)
#print(neighbour1)
pop_list = range(1,len(comb1)+1)
for i in pop_list:
    neighbour1.pop(i)
#print("B_233 dopant and neighbours and distance:",neighbour1)
dopant_bond_a = neighbour1[0][-1]+neighbour1[1][-1]
dopant_bond_b = neighbour1[2][-1]+neighbour1[3][-1]
dopant_bond_c = neighbour1[4][-1]+neighbour1[5][-1]
average_dopant_bond_a = dopant_bond_a/2
average_dopant_bond_b = dopant_bond_b/2
average_dopant_bond_c = dopant_bond_c/2
distance1 = []
for i in neighbour1:
    distance1.append(i[-1])
average_dopant_bond = sum(distance1)/len(distance1)
print("\n","B_233 average_dopant_bond_a:",average_dopant_bond_a,"\n","B_233 average_dopant_bond_b:",average_dopant_bond_a,"\n","B_233 average_dopant_bond_c:",average_dopant_bond_c,"\n","B_233 average_dopant_bond:",average_dopant_bond)

###To determine the adjacent Pb and its I pair and bond distance
B_233_comb = [ ]
for i in B_233:
    B_233_comb.append(tuple(i))
#print("B_233 adjacent combinations:",B_233_comb)

comb2 = [ ]
for i in B_233_comb:
   # comb2 = [ ]
    center_Pb = i[0]
    coordination_I = i[1]
    #print("center_Pb:",center_Pb,"\n","coordination_I:",coordination_I)
    for j in coordination_I:
        temp = []
        temp.append(center_Pb)
        temp.append(j)
       # print()
        comb2.append(temp)
#print("B_233 adjacent Pb and neighbour combination:",comb2)

neighbour2 = []
for i in comb2:
    for j in connec.get_connections():
        if list_a_is_in_list_b(i,j):
            #print(j)
            neighbour2.append(j)
# To eliminate the redundant pairs
pop_list = range(1,len(comb2)+1)
for i in pop_list:
    neighbour2.pop(i)
#print("B_233 adjacent Pb and neighbour combination and distances:",neighbour2)

distance_adjacent_a = [ ]
distance_adjacent_b = [ ]
distance_adjacent_c = [ ]
a2  = list(range(0,len(neighbour2),6)) + list(range(1,len(neighbour2),6))
b2  = list(range(2,len(neighbour2),6)) + list(range(3,len(neighbour2),6))
c2  = list(range(4,len(neighbour2),6)) + list(range(5,len(neighbour2),6)) 
for i in a2:
    distance_adjacent_a.append(neighbour2[i][-1])
for i in b2:
    distance_adjacent_b.append(neighbour2[i][-1])
for i in c2:
    distance_adjacent_c.append(neighbour2[i][-1])
average_distance_adjacent_a = sum(distance_adjacent_a)/len(distance_adjacent_a)
average_distance_adjacent_b = sum(distance_adjacent_b)/len(distance_adjacent_b)
average_distance_adjacent_c = sum(distance_adjacent_c)/len(distance_adjacent_c)
average_distance_adjacent = sum(distance_adjacent_a+distance_adjacent_b+distance_adjacent_c)/len(neighbour2)
print("B_233 average_distance_adjacent_a:",average_distance_adjacent_a)
print("B_233 average_distance_adjacent_b:",average_distance_adjacent_b)
print("B_233 average_distance_adjacent_c:",average_distance_adjacent_c)
print("B_233 average_distance_adjacent:",average_distance_adjacent)

print("B_233 average_dopant_bond_a-average_distance_adjacent_a:",average_dopant_bond_a-average_distance_adjacent_a)
print("B_233 average_dopant_bond_b-average_distance_adjacent_b:",average_dopant_bond_b-average_distance_adjacent_b)
print("B_233 average_dopant_bond_c-average_distance_adjacent_c:",average_dopant_bond_c-average_distance_adjacent_c)
print("B_233 average_dopant_bond-average_distance_adjacent:",average_dopant_bond-average_distance_adjacent)






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


