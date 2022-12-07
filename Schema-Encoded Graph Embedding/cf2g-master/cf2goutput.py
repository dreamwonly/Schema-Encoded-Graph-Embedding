import os
import json

import graph
import networkx as nx
from collections import defaultdict

#######################
#    output funcs     #
#######################
def one_hot_encode(idx, len):
    ret = [0] * len
    ret[idx] = 1
    return ret

#output a JSON-formatted dictionary where key = node type, value = list of nodes
def output_node_types(G, output_path=None):
    print("outputting node types to JSON format...")

    type_dict = {}
    types = list(G.nodetypes.keys())
    for node in G.nodes:
        type_dict[G.nodes[node].id()] = one_hot_encode(types.index(G.nodes[node].getType()), len(types))
    with open(f'{output_path}', 'w') as f:
        f.write(str(types))
        json.dump(type_dict, f, indent=2)
    print("Node types outputted to " + output_path + ".")


#output a JSON-formatted dictionary where key = edge type, value = list of nodes
def output_edge_types(G, output_path=None):
    print("outputting edge types to JSON format...")

    type_dict = {}
    types = list(G.edgetypes.keys())
    for edge in G.edges:
        type_dict[G.edges[edge].id()] = one_hot_encode(types.index(G.edges[edge].getType()), len(types))

    with open(f'{output_path}', 'w') as f:
        f.write(str(types))
        json.dump(type_dict, f, indent=2)
    print("Edge types outputted to " + output_path + ".")

# output a JSON-formatted dictionary where key = edge_schema, value = src node list and dest node list where src[i]-edge-dest[i]
def output_graph(G, output_path=None):
    print("outputting graph to JSON format...")

    output_dict = defaultdict(list)
    for schema in G.get_schemas():
        str_schema = '-'.join(schema)
        src_dst = G.get_schema_node_lists(schema)
        # print(src_dst)
        output_dict[str_schema] = (src_dst[0], src_dst[1])

    if not output_path:
        output_path = os.getcwd() + "/output/graph.json"
    with open(f'{output_path}', 'w') as f:
        json.dump(output_dict, f)
    print("Graph outputted to " + output_path + ".")

# serialize the graph with a default path name.
def pickle_graph(G, output_path=None):
    print("converting graph to pickle...")
    if not output_path:
        output_path = os.getcwd() + "/output/graph.gpickle"
    nx.write_gpickle(G, output_path)
    print("Graph pickle outputted to " + output_path + ".")

def output_node_features(G, output_path=None):
    print("outputting node features...")

    feature_dict = {}
    hashlist = list(G.nodehashes)

    for node in G.nodes:
        if G.nodes[node].hasFeatures():
            bucket = one_hot_encode(hashlist.index(G.nodes[node].features["hash"]), len(hashlist))
            feature_dict[G.nodes[node].id()] = bucket

    if not output_path:
        output_path = os.getcwd() + "/output/node_features.json"
    with open(f'{output_path}', 'w') as f:
        json.dump(feature_dict, f)
    print("Node features outputted to " + output_path + ".")

def output_edge_features(G, output_path=None):
    print("outputting edge features... and ouput edge timestamp features")
    output_time_path = os.getcwd() + "/output/edge_time_features.json"
    feature_dict = {}
    time_feature_dict = {}
    hashlist = list(G.edgehashes)
    for edge in G.edges:
        if G.edges[edge].hasFeatures():
            bucket = one_hot_encode(hashlist.index(G.edges[edge].features["hash"]), len(hashlist))
            feature_dict[G.edges[edge].id()] = bucket
            time_feature_dict[G.edges[edge].id()] = G.edges[edge].features["time"]

    if not output_path:
        output_path = os.getcwd() + "/output/edge_features.json"
        
    with open(f'{output_path}', 'w') as f:
        json.dump(feature_dict, f)
    with open(f'{output_time_path}', 'w') as f:
        json.dump(time_feature_dict, f)
    print("Edge features outputted to " + output_path + ".")
