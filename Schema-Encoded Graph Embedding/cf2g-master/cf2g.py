import cf2goutput
import errors as cf2gerror
import filters
import utils
import graph

import networkx as nx

import argparse
import os
import sys

#############
#   utils   #
#############
#def draw_graph(output_path=None):
#    print("Making NetworkX Graph...")
#    G = nx.MultiDiGraph()
#    G.add_nodes_from(get_nodes_with_labels())
#    G.add_edges_from(get_edges_with_labels())
#    print("Graph constructed.")
#    edge_labels=nx.draw_networkx_edge_labels(G,pos=nx.spring_layout(G))
#    print("Drawing graph...")
#    if not output_path:
#        output_path = os.getcwd() + "/output/graph.png"
#    A = nx.nx_agraph.to_agraph(G)
#    A.draw(output_path, format="png", prog="sfdp", args="-Goverlap=scale")
#    print("Graph drawn to " + output_path + ".")

##############
#   Output   #
##############

def print_graph_info(G):
        print("number of nodes: " + str(G.num_nodes()))
        print("number of node types: " + str(G.num_node_types()))
        for type in G.nodetypes:
            print(type + " - " + str(len(G.nodetypes[type])))
        #for node in G.get_nodes_with_types():
        #    print(str(node[1]) + " " + str(node[0]))
        print("number of edges: " + str(G.num_edges()))
        print("number of edge types: " + str(G.num_edge_types()))
        for type in G.edgetypes:
            print(type + " - " + str(len(G.edgetypes[type])))
        #for edge in G.get_edges_with_types():
        #    print(str(edge[0][1]) + " " + str(edge[0][0]) + " ---> " + str(edge[1][1]) + " " + str(edge[1][0]))
        print("number of distinct edge schemas: " + str(G.num_schemas()))
        for schema in G.get_schemas():
            print(schema)

def print_graph_info_to_file(G, output_path=None):
    if not output_path:
        output_path = os.getcwd() + "/output/graph_stats.txt"
    with open(f'{output_path}', 'w') as f:
        f.write("number of nodes: " + str(G.num_nodes()) + "\n")
        f.write("number of node types: " + str(G.num_node_types()) + "\n")
        for type in G.nodetypes:
            f.write(type + " - " + str(len(G.nodetypes[type])) + "\n")
        #for node in G.get_nodes_with_types():
        #    print(str(node[1]) + " " + str(node[0]))
        f.write("number of edges: " + str(G.num_edges()) + "\n")
        f.write("number of edge types: " + str(G.num_edge_types()) + "\n")
        for type in G.edgetypes:
            f.write(type + " - " + str(len(G.edgetypes[type])) + "\n")
        #for edge in G.get_edges_with_types():
        #    print(str(edge[0][1]) + " " + str(edge[0][0]) + " ---> " + str(edge[1][1]) + " " + str(edge[1][0]))
        f.write("number of distinct edge schemas: " + str(G.num_schemas()) + "\n")
        for schema in G.get_schemas():
            str_schema = '-'.join(schema)
            f.write(str(str_schema) + "\n")


##########################
#  Construction Library  #
##########################

###############
#  NetworkX   #
###############

# make a graph with whatever has been loaded into our storage.
def create_networkx_graph_with_attributes(G):
    print("Making NetworkX Graph...")
    nxG = nx.MultiDiGraph()
    nxG.add_nodes_from(G.get_nodes_with_attributes())
    nxG.add_edges_from(G.get_edges_with_attributes())
    print("Graph constructed.")
    return nxG

def create_networkx_graph(G):
    print("Making NetworkX Graph...")
    nxG = nx.MultiDiGraph()
    nxG.add_nodes_from(G.get_nodes())
    nxG.add_edges_from(G.get_edges())
    print("Graph constructed.")
    return nxG

# serialize the graph with a default path name.
def pickle_graph(G, output_path=None):
    print("converting graph to pickle...")
    if not output_path:
        output_path = os.getcwd() + "/output/graph.gpickle"
    nx.write_gpickle(create_networkx_graph_with_attributes(G), output_path)
    print("Graph pickle outputted to " + output_path + ".")

#############
#    DGL    #
#############

# make a graph with whatever has been loaded into our storage
#def create_dgl_graph_with_attributes(G):

#def create_dgl_graph(G):




###############
#    Main     #
###############
parser = argparse.ArgumentParser()

#python3 main.py -f w3cprov -i /home/user/input.json -g /home/user/graph.json
parser.add_argument("-f", "--filter", help="Data input format (options are: w3cprov, spade).", required=True)
parser.add_argument("-i", "--input", help="Absolute path for the data input file.", required=True)
parser.add_argument("-g", "--graph", help="Output the graph JSON to the provided path.")
parser.add_argument("-p", "--pickle", help="Output the pickle graph to the provided path.")
parser.add_argument("-s", "--stats", help="Output the stats file with node/edge types and indexes to the provided path.")
parser.add_argument("-d", "--draw", help="Output drawn graph to the provided path.")
parser.add_argument("-ng", "--nodegranularity", help="Configure the granularity of the edge types to be 'coarse' or 'fine' (w3c only).")
parser.add_argument("-eg", "--edgegranularity", help="Configure the granularity of the edge types to be 'coarse' or 'fine' (w3c only).")
parser.add_argument("-nf", "--nodefeatures", help="Output the features files with node/edge features to the provided path.")
parser.add_argument("-ef", "--edgefeatures", help="Output the features files with node/edge features to the provided path.")
parser.add_argument("-nt", "--nodetypes", help="Output the node type feature file to the provided path.")
parser.add_argument("-et", "--edgetypes", help="Output the edge type feature file to the provided path.")
args = parser.parse_args()
G = graph.Graph()

try:
    f = open(args.input, 'r')
except OSError:
    print("Could not open/read file: ", args.input, ", are you sure you provided the correct path?")
    sys.exit()

if args.filter == 'spade':
    filter = filters.SpadeFilter()
    filter.load_from_file(args.input, G)

elif args.filter == 'w3cprov':
    filter = filters.W3CFilter()
    filter.load_from_file(args.input, G, args.nodegranularity, args.edgegranularity)
elif args.filter == 'streamspot':
    filter = filters.StreamSpotFilter()
    filter.load_from_file(args.input, G)
else:
    cf2gerror.format_unrecognized()

if args.graph:
    cf2goutput.output_graph(G, args.graph)

if args.pickle:
    NxG = create_networkx_graph_with_attributes(G)
    cf2goutput.pickle_graph(NxG, args.pickle)

if args.stats:
    print_graph_info_to_file(G, args.stats)

if args.nodefeatures:
    cf2goutput.output_node_features(G, args.nodefeatures)

if args.edgefeatures:
    cf2goutput.output_edge_features(G, args.edgefeatures)

if args.nodetypes:
    cf2goutput.output_node_types(G, args.nodetypes)

if args.edgetypes:
    cf2goutput.output_edge_types(G, args.edgetypes)

if args.draw:
    draw_graph(args.draw)
