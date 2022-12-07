import json
import os
import sys
import time

import graph
import utils

def one_hot_encode(idx, len):
    ret = [0] * len
    ret[idx] = 1
    return ret
"""
When inputting data from CamFlow, a filter can be applied to parse either
Spade format or W3C format.
"""

class SpadeFilter():
    """
    Initialize the filter object
    """
    def __init__(self):
        self.loaded = False

    """
    Load the specified graph from file, using the filter to parse the data.
    params: input_path, the full file path to the camflow data.
    params: G, the cf2g graph object we are loading to
    """
    def load_from_file(self, input_path, G):
        f = open(input_path, 'r', errors='replace')
        objects = f.readlines()
        print("Loading data now...")
        start = time.time()
        linecount = 0

        for object in objects:
            obj = json.loads(object)
            linecount = linecount + 1
            if "from" not in obj:
                self.load_node(obj, G)
            else:
                self.load_edge(obj, G)
            if linecount % 100000 == 0:
                end = time.time()
                hours, rem = divmod(end-start, 3600)
                minutes, seconds = divmod(rem, 60)
                print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
                print("loaded " + str(round(((linecount/len(objects))*100), 2)) + "% of data in " +
                      ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)) + " seconds.")
                start = time.time()
        self.loaded = True
        print("Data loaded! Added " + str(linecount) + " lines.")

    """
    This is a helper function to load the edge found in a spade file.
    params: object, the JSON object to parse the edge from.
    params: G, the cf2g graph object that the edge will be loaded to.
    """
    def load_edge(self, object, G):
        edge_type = object["type"]
        src_node = object["from"]
        dst_node = object["to"]
        features = object["annotations"]
        src_node_type = features["from_type"]
        dst_node_type = features["to_type"]
        edge = features["id"]
        time = features["jiffies"]

        G.add_node(src_node, src_node_type)
        G.add_node(dst_node, dst_node_type)
        G.add_edge(edge, edge_type, src_node, src_node_type, dst_node, dst_node_type)
        self.load_features(edge_type, time, features, edge, False, G)

    """
    This is a helper function to load the node found in a spade file.
    params: object, the JSON object to parse the node from.
    params: G, the cf2g graph object that the node will be loaded to.
    """
    def load_node(self, object, G):
        node = object["id"]
        features = object["annotations"]
        node_type = features["object_type"]
        try:
            time = features["cf:jiffies"]
        except:
            time = 0

        G.add_node(node, node_type)
        self.load_features(node_type, time, features, node, True, G)

    """
    This is a helper function to load the features found in a spade file for node or edge.
    params: type, the node or edge type which we use as a feature.
    params: time, the jiffies count which we use as a feature.
    params; features, the features which cf writes given as a dictionary.
    params: serial, the cf2g given name for the node or edge.
    params: isNode, true if this is a node serial, false if edge.
    params: G, the cf2g graph object that the features will be loaded to.
    """
    def load_features(self, type, time, features, serial, isNode, G):
        features_dict = {}
        hash = utils.hash_features(features)
        features_dict["type"] = type
        features_dict["time"] = time
        features_dict["hash"] = hash

        if isNode:
            G.nodes[serial].add_features(features_dict)
            G.nodehashes.add(hash)
        else:
            G.edges[serial].add_features(features_dict)
            G.edgehashes.add(hash)

class W3CFilter():
    """
    Initialize the filter.
    """
    def __init__(self):
        self.loaded = False

    """
    Load the specified graph from file, using the filter to parse the data.
    params: input_path, the full file path to the camflow data.
    params: G, the cf2g graph object we are loading to
    """
    def load_from_file(self, input_path, G, node_gran, edge_gran):
        json_data = []
        with open(input_path, 'r') as f:
            for line in f:
        # print(data)
                json_data.append(json.loads(line))

        if node_gran is None or node_gran != "coarse" and node_gran != "fine":
            node_gran = "coarse"
        if edge_gran is None or edge_gran != "coarse" and edge_gran != "fine":
            edge_gran = "coarse"

        print("Loading data now...")
        logcount = 0
        start = time.time()

        for object in json_data:
            logcount += 1
            features = {}
            jiffies = ""
            type = ""

            for prov_type in object:

                if prov_type == "activity":
                    for node in object[prov_type]:
                        features = object[prov_type][node]
                        jiffies = object[prov_type][node]["cf:jiffies"]
                        if node_gran == "fine":
                            type = object[prov_type][node]["prov:type"]
                        else:
                            type = prov_type
                        self.load_node(node, type, G)
                        self.load_features(type, jiffies, features, node, True, G)

                elif prov_type == "entity":
                    for node in object[prov_type]:
                        if "cf:camflow" in object[prov_type][node]:
                            continue
                        features = object[prov_type][node]
                        jiffies = object[prov_type][node]["cf:jiffies"]
                        if node_gran == "fine":
                            type = object[prov_type][node]["prov:type"]
                        else:
                            type = prov_type
                        self.load_node(node, type, G)
                        self.load_features(type, jiffies, features, node, True, G)

                elif prov_type == "agent":
                    for node in object[prov_type]:
                        features = object[prov_type][node]
                        jiffies = object[prov_type][node]["cf:jiffies"]
                        if node_gran == "fine":
                            type = object[prov_type][node]["prov:type"]
                        else:
                            type = prov_type
                        self.load_node(node, type, G)
                        self.load_features(type, jiffies, features, node, True, G)

                elif prov_type == "prefix":
                    continue
                else:
                    for edge in object[prov_type]:
                        features = object[prov_type][edge]
                        jiffies = object[prov_type][edge]["cf:jiffies"]
                        if edge_gran == "fine":
                            type = object[prov_type][node]["prov:type"]
                        else:
                            type = prov_type
                        self.load_edge(edge, type, prov_type, object[prov_type][edge], G)
                        self.load_features(type, jiffies, features, edge, False, G)

                if logcount % 1000 == 0:
                    end = time.time()
                    hours, rem = divmod(end-start, 3600)
                    minutes, seconds = divmod(rem, 60)
                    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
                    print("loaded " + str(round(((logcount/len(json_data))*100), 2)) + "% of data in " +
                          ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)) + " seconds.")
                    start = time.time()
        self.loaded = True
        print("Data loaded! Added " + str(logcount) + " logs.")

    """
    This is a helper function to load the node found in a w3c file.
    params: node, the cf2g given serial name.
    params: node_type, the cf node type (file, process_memory, etc.)
    params: G, the cf2g graph object that the node will be loaded to.
    """
    def load_node(self, node, node_type, G):
        G.add_node(node, node_type)

    """
    This is a helper function to load the edge found in a w3cprov file.
    params: edge, the JSON object to parse the edge from.
    params: prov_type, indicates the w3cprov relation type.
    params: values, JSON object to parse the src and dst node from.
    params: G, the cf2g graph object that the edge will be loaded to.
    """
    def load_edge(self, edge, type, prov_type, values, G):
        src_node = ""
        dst_node = ""

        if prov_type == "wasGeneratedBy":
            src_node = values["prov:entity"]
            dst_node = values["prov:activity"]
        elif prov_type == "used":
            src_node = values["prov:activity"]
            dst_node = values["prov:entity"]
        elif prov_type == "wasInformedBy":
            src_node = values["prov:informant"]
            dst_node = values["prov:informed"]
        elif prov_type == "wasInfluencedBy":
            src_node = values["prov:influencee"]
            dst_node = values["prov:influencer"]
        elif prov_type == "wasAssociatedWith":
            src_node = values["prov:agent"]
            dst_node = values["prov:activity"]
        elif prov_type == "wasDerivedFrom":
            src_node = values["prov:generatedEntity"]
            dst_node = values["prov:usedEntity"]
        # not currently supported by CamFlow
        elif prov_type == "wasStartedBy" or prov_type == "wasEndedBy":
            src_node = values["prov:trigger"]
            dst_node = values["prov:activity"]
        elif prov_type == "wasInvalidatedBy":
            src_node = values["prov:activity"]
            dst_node = values["prov:entity"]
        elif prov_type == "wasAttributedTo":
            src_node = values["prov:entity"]
            dst_node = values["prov:agent"]
        elif prov_type == "actedOnBehalfOf":
            src_node = values["prov:delegate"]
            dst_node = values["prov:responsible"]
        elif prov_type == "specializationOf":
            src_node = values["prov:specificEntity"]
            dst_node = values["prov:generalEntity"]
        else:
            cf2gerror.relation_error()
            sys.exit()

        if src_node in G.nodes and dst_node in G.nodes:
            G.add_edge(edge, type, src_node, G.nodes[src_node].getType(), dst_node, G.nodes[dst_node].getType())

    """
    This is a helper function to load the features found in a w3cprov file for node or edge.
    params: type, the node or edge type which we use as a feature.
    params: time, the jiffies count which we use as a feature.
    params; features, the features which cf writes given as a dictionary.
    params: serial, the cf2g given name for the node or edge.
    params: isNode, true if this is a node serial, false if edge.
    params: G, the cf2g graph object that the features will be loaded to.
    """
    def load_features(self, type, time, features, serial, isNode, G):
        features_dict = {}
        hash = utils.hash_features(features)
        features_dict["type"] = type
        features_dict["time"] = time
        features_dict["hash"] = hash

        if isNode:
            if serial in G.nodes:
                G.nodes[serial].add_features(features_dict)
                G.nodehashes.add(hash)
        else:
            if serial in G.edges:
                G.edges[serial].add_features(features_dict)
                G.edgehashes.add(hash)



class StreamSpotFilter():
    """
    Initialize the filter object
    """
    def __init__(self):
        self.loaded = False

    """
    Load the specified graph from file, using the filter to parse the data.
    params: input_path, the full file path to the camflow data.
    params: G, the cf2g graph object we are loading to
    """
    def load_from_file(self, input_path, G):
        # map for one hot encode

        node_type_length = 8
        edge_type_length = 26
        map = {'a' : 1,
        'b' : 2,
       'c': 3,
       'd': 4,
       'e': 5,
       'f': 1,
       'g': 2,
       'h': 3,
       'i': 4,
       'j': 5,
       'k': 6,
       'l': 7,
       'm': 8,
       'n': 9,
       'o': 10,
       'p': 11,
       'q': 12,
       'r': 13,
       's': 14,
       't': 15,
       'u': 16,
       'v': 17,
       'w': 18,
       'x': 19,
       'y': 20,
       'z': 21,
       'A': 22,
       'B': 23,
       'C': 24,
       'D': 25,
       'E': 26,
       'F': 27,
       'G': 28,
       'H': 29,
      }

        f = open(input_path, 'r', errors='replace')
        objects = f.readlines()
        print("Loading data now...")
        start = time.time()
        linecount = 0
        edge_id = -1
        for line in objects:
            # obj = json.loads(object)
            obj = dict()
            src, src_class, dst, dst_class, edge_class, graph_id = line.split('\t')
            edge_id = edge_id + 1
            linecount = linecount + 1
            # if "from" not in obj:
            #     self.load_node(obj, G)
            # else:
            #     self.load_edge(obj, G)
            # src_node_feature = one_hot_encode(map[src_class]-1, node_type_length)
            # dst_node_feature = one_hot_encode(map[dst_class]-1, node_type_length)
            # edge_feature = one_hot_encode(map[edge_class]-1, edge_type_length)

            obj["from"] = src
            obj["to"] = dst
            obj["id"] = edge_id
            obj["from_type"] = src_class
            obj["to_type"] = dst_class
            obj["type"] = edge_class

            # obj["from_feat"] = src_node_feature
            # obj["to_feat"] = dst_node_feature
            # obj["edge_feat"] = edge_feature

            self.load_edge(obj, G)

            if linecount % 10000 == 0:
                end = time.time()
                hours, rem = divmod(end-start, 3600)
                minutes, seconds = divmod(rem, 60)
                print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
                print("loaded " + str(round(((linecount/len(objects))*100), 2)) + "% of data in " +
                      ("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)) + " seconds.")
                start = time.time()
        self.loaded = True
        print("Data loaded! Added " + str(linecount) + " lines.")

    """
    This is a helper function to load the edge found in a spade file.
    params: object, the JSON object to parse the edge from.
    params: G, the cf2g graph object that the edge will be loaded to.
    """
    def load_edge(self, object, G):
        
        src_node = object["from"]
        dst_node = object["to"]
        edge = object["id"]

        # features = object["annotations"]
        src_node_type = object["from_type"]
        dst_node_type = object["to_type"]
        edge_type = object["type"]

        # edge = features["id"]
        # time = features["jiffies"]
        # edge_feature = object["edge_feat"]
        # dst_node_feature = object["to_feat"] 
        # src_node_feature = object["from_feat"] 

        G.add_node(src_node, src_node_type)
        G.add_node(dst_node, dst_node_type)
        # print(edge, edge_type, src_node, src_node_type, dst_node, dst_node_type)
        G.add_edge(edge, edge_type, src_node, src_node_type, dst_node, dst_node_type)
        # for schema in G.get_schemas():
        #     src_dst = G.get_schema_node_lists(schema)
        #     print(src_dst)
        # self.load_features(edge_feature, edge, False, G)
        # self.load_features(src_node_feature, src_node, True, G)
        # self.load_features(dst_node_feature, dst_node, True, G)

    """
    This is a helper function to load the node found in a spade file.
    params: object, the JSON object to parse the node from.
    params: G, the cf2g graph object that the node will be loaded to.
    """
    # def load_node(self, object, G):
    #     node = object["id"]
    #     features = object["annotations"]
    #     node_type = features["object_type"]
    #     try:
    #         time = features["cf:jiffies"]
    #     except:
    #         time = 0

    #     G.add_node(node, node_type)
    #     self.load_features(node_type, time, features, node, True, G)

    """
    This is a helper function to load the features found in a spade file for node or edge.
    params: type, the node or edge type which we use as a feature.
    params: time, the jiffies count which we use as a feature.
    params; features, the features which cf writes given as a dictionary.
    params: serial, the cf2g given name for the node or edge.
    params: isNode, true if this is a node serial, false if edge.
    params: G, the cf2g graph object that the features will be loaded to.
    """
    def load_features(self, feat, serial, isNode, G):
        # features_dict = {}
        # hash = utils.hash_features(features)
        # features_dict["type"] = type
        # features_dict["time"] = time
        # features_dict["hash"] = hash

        if isNode:
            G.nodes[serial].add_features(feat)
            # G.nodehashes.add(hash)
        else:
            G.edges[serial].add_features(feat)
            # G.edgehashes.add(hash)
