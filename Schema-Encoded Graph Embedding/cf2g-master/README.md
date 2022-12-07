# CF2G

"CamFlow 2 Graph" (CF2G) is a Python library for converting CamFlow provenance data into JSON files or NetworkX graphs.
Graphs can be used directly or serialized/exported via pickle.
CF2G works as a command line tool or as an imported library.

## Installation

NetworkX is a required dependency. For graph drawing, pydot, graphviz, and matplotlib are required dependencies.

```bash
pip install networkx
pip install pydot
pip install matplotlib
brew install graphviz
```

## Primary Usage

```python
import cf2g

# 1) load SPADE JSON input
cf2g.load_spade_file(os.getcwd() + "/input/my_file.json") # load file from input path

#OR

# 2) load W3C PROV JSON format
cf2gloader.load_w3c_file(os.getcwd() + "/input/sample.json") # load file from input path

# 2) create graph with attributes
G = cf2g.create_graph() # you can use the graph G directly from here

# 2.5) png graph
cf2gwriter.draw_graph_matplotlib(G, "/output/my_graph.png") # use matplotlib and make graph to provided path
cf2gwriter.draw_graph_dot(G, "/output/my_graph.png") # use graphviz and make graph to provided path

# 3) serialize using pickle
cf2g.pickle_graph(G, "/output/my_graph.gpickle") # serialize to the provided path

# 4) output to json
cf2gwriter.output_graph() #output graph with edge schemas to provided path or default
cf2gwriter.output_node_types() #output node type list to provided path or default
cf2gwriter.output_node_feature_count() #output nodes with their instance and feature count to provided path or default
cf2gwriter.output_edge_feature_count() #output edges with their instance and feature count to provided path or default

cf2g.clear() # resets internal database/dictionaries
```

## Command Line Usage
```bash
usage: cf2grun.py [-h] -f FORMAT -i INPUT [-g] [-p] [-nf] [-ef] [-nt]

optional arguments:
  -h, --help            show this help message and exit
  -f FORMAT, --format FORMAT
                        Data input format (options are: w3cprov, spade).
  -i INPUT, --input INPUT
                        Absolute path for the data input file.
  -g, --graph           Output the graph JSON to the default path.
  -p, --pickle          Output the pickle graph to the default path.
  -nf, --nodefeatures   Output the node features to the default path.
  -ef, --edgefeatures   Output the edge features to the default path.
  -nt, --nodetypes      Output the node type list to the default path.

```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
