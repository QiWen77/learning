from pymatgen.core.structure import IStructure,Structure
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import ValenceIonicRadiusEvaluator,VoronoiNN
from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder,JMolCoordFinder,VoronoiAnalyzer,RelaxationAnalyzer,VoronoiConnectivity
s1 = Structure.from_file('POSCAR.orig')
s2 = Structure.from_file('CONTCAR.relax2')
Voronoi = VoronoiCoordFinder(s2,target=None)
# site = s2.num_sites
# print s2[0]
polyhedra = Voronoi.get_voronoi_polyhedra(1)
coordinate = Voronoi.get_coordination_number(1)
coordinate_sites = Voronoi.get_coordinated_sites(1)
Voronoi_Analyzer = VoronoiAnalyzer()
anay = Voronoi_Analyzer.analyze(s1,n=0)
strucs = [s1,s2]
anays = Voronoi_Analyzer.analyze_structures(strucs)
print anay
# plt = Voronoi_Analyzer.plot_vor_analysis(anays)
relax = RelaxationAnalyzer(s1,s2)
volume_change = relax.get_percentage_volume_change()
lattice_parameter_changes = relax.get_percentage_lattice_parameter_changes()
bond_dist = relax.get_percentage_bond_dist_changes()
connec = VoronoiConnectivity(s2)
# print dir(connec)
# max_bond = connec.get_max_bond_lengths(s2)
# print volume_change,lattice_parameter_changes,bond_dist
valence =  BVAnalyzer().get_valences(s2)
oxide_state = BVAnalyzer().get_oxi_state_decorated_structure(s2)
# print "percent volume change(0.055 implies a 5.5% increase):",volume_change
# print "percent lattice change(-0.055 implies a 5.5% decrease):",lattice_parameter_changes
# print "percent bond distance change(0.055 implies a 5.5% increase):",bond_dist[1]
# print "valences of structure:",valence
# print "oxidation states:",oxide_state
VIRE = ValenceIonicRadiusEvaluator(s2)
radii = VIRE.radii
get_radii = VIRE._get_ionic_radii()
ox_structure = VIRE._get_valences()
# print ox_structure
# target = ["I","Pb","K","C","H","N"]
NN = VoronoiNN(targets=None)
cn = NN.get_nn_info(s2,0)
poly = NN.get_voronoi_polyhedra(s2,0)
# nn = NN.get_nn(s2,0)
# wnn = NN.get_weights_of_nn_sites(s2,0)
# img = NN.get_nn_images(s2,0)
nn_info = NN.get_nn_info(s2,0)
print poly#,nn_info

