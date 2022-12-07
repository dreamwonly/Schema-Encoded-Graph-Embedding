from collections import defaultdict
from errors import *

import utils

class Node():
    def __init__(self, serial, type, index):
        self.index = index
        # node type
        self.type = type
        # features of the node
        self.features = None
        # cf given name
        self.serial = serial

    def id(self):
        return self.index

    def getType(self):
        return self.type

    def serial(self):
        return self.serial

    def hasFeatures(self):
        return self.features != None

    def getFeatures(self):
        return self.features

    def add_features(self, features):
        self.features = features

class Edge():
    def __init__(self, serial, type, src_node, dst_node, index):
        self.index = index
        # edge type
        self.type = type
        # features of the edge
        self.features = None
        # source node of the edge
        self.src_node = src_node
        # dest node of the edge
        self.dst_node = dst_node
        # cf given name
        self.serial = serial

    def id(self):
        return self.index

    def serial(self):
        return self.serial

    def getType(self):
        return self.type

    def getSrcNode(self):
        return self.src_node

    def getDstNode(self):
        return self.dst_node

    def hasFeatures(self):
        return self.features != None

    def getFeatures(self):
        return self.features

    def add_features(self, features):
        self.features = features

class Graph():
    def __init__(self):
        # node serial to Node object
        self.nodes = defaultdict()
        # edge serial to Edge object
        self.edges = defaultdict()
        # type schema (nt-et-nt), to list of node index tuples (src,dst)
        self.schemas = defaultdict(list)
        self.nodetypes = defaultdict(list)
        self.edgetypes = defaultdict(list)
        self.nodehashes = set()
        self.edgehashes = set()

    def num_nodes(self):
        return len(self.nodes)

    def num_edges(self):
        return len(self.edges)

    def num_node_types(self):
        return len(self.nodetypes)

    def num_edge_types(self):
        return len(self.edgetypes)

    def num_schemas(self):
        return len(self.schemas)

    def num_nodes_of_type(self, type):
        if type in self.nodetypes:
            return len(self.nodetypes[type])
        return 0

    def num_edges_of_type(self, type):
        if type in self.edgetypes:
            return len(self.edgetypes[type])
        return 0

    def num_edges_of_schema(self, schema):
        if schema in self.schemas:
            return len(self.schemas[schema])
        return 0

    #returns list of node ids
    def get_nodes(self):
        node_list = []
        for node in self.nodes:
            node_list.append(self.nodes[node].id())
        return node_list

    def get_node_types(self):
        return self.nodetypes

    def get_nodes_with_types(self):
        node_list = []
        for node in self.nodes:
            node_list.append((self.nodes[node].id(), self.nodes[node].getType()))
        return node_list

    def get_nodes_with_attributes(self):
        node_list = []
        for node in self.nodes:
            node_list.append((self.nodes[node].id(), self.nodes[node].getFeatures()))
        return node_list

    def get_edges(self):
        edge_list = []
        for edge in self.edges:
            edge_list.append((self.nodes[self.edges[edge].getSrcNode()].id(),
                             self.nodes[self.edges[edge].getDstNode()].id()))
        return edge_list

    def get_edges_with_types(self):
        edge_list = []
        for edge in self.edges:
            edge_list.append(((self.nodes[self.edges[edge].getSrcNode()].id(),
                               self.nodes[self.edges[edge].getSrcNode()].getType()),
                              (self.nodes[self.edges[edge].getDstNode()].id(),
                               self.nodes[self.edges[edge].getDstNode()].getType())))
        return edge_list

    def get_edges_with_attributes(self):
        edge_list = []
        for edge in self.edges:
            edge_list.append((self.nodes[self.edges[edge].getSrcNode()].id(),
                              self.nodes[self.edges[edge].getDstNode()].id(),
                              self.edges[edge].getFeatures()))
        return edge_list

    def get_schemas(self):
        return list(self.schemas.keys())

    def get_schema_node_lists(self, schema):
        return list(map(list, zip(*self.schemas[schema])))

    def add_node(self, serial, type):
        self.nodes[serial] = Node(serial, type, int(serial)) # len() 改为 int(serial) for streamspot
        self.nodetypes[type].append(serial)

    def add_edge(self, serial, type, src_node, src_node_type, dst_node, dst_node_type):
        self.edges[serial] = Edge(serial, type, src_node, dst_node, len(self.edges))
        self.edgetypes[type].append(serial)

        # add nodes
        if src_node not in self.nodes:
            self.nodes[serial] = Node(src_node, src_node_type, len(self.nodes))
            self.nodetypes[src_node_type].append(src_node)
        if dst_node not in self.nodes:
            self.nodes[serial] = Node(dst_node, dst_node_type, len(self.nodes))
            self.nodetypes[dst_node_type].append(dst_node)

        # add edge schema
        self.schemas[(src_node_type, type, dst_node_type)].append((self.nodes[src_node].id(), self.nodes[dst_node].id()))
        # print(self.schemas)

    def clear(self):
        self.nodes = defaultdict()
        self.edges = defaultdict()
        self.schemas = defaultdict(list)
        self.nodetypes = defaultdict(list)
        self.edgetypes = defaultdict(list)
