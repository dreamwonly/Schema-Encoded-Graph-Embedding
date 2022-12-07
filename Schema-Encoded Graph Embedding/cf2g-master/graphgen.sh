#!/bin/bash

CURR=121
INPUTDIR="$PWD/input"
OUTPUTDIR="$PWD/output"

for in in $INPUTDIR/*.log
  do
    python3 cf2g.py -f w3cprov -i $in \
                             -ng "fine" \
                             -eg "coarse" \
                             -g "$OUTPUTDIR/graph$CURR.json" \
                             -s "$OUTPUTDIR/graph_stats$CURR.txt" \
                             -p "$OUTPUTDIR/graph$CURR.gpickle" \
                             -nf "$OUTPUTDIR/node_features_graph$CURR.json" \
                             -ef "$OUTPUTDIR/edge_features_graph$CURR.json" \
                             -nt "$OUTPUTDIR/node_types_graph$CURR.json" \
                             -et "$OUTPUTDIR/edge_types_graph$CURR.json"

    ln "$OUTPUTDIR/graph$CURR.json" "graph$CURR.json"
    ln "$OUTPUTDIR/node_features_graph$CURR.json" "node_features_graph$CURR.json"
    ln "$OUTPUTDIR/edge_features_graph$CURR.json" "edge_features_graph$CURR.json"
    ln "$OUTPUTDIR/node_types_graph$CURR.json" "node_types_graph$CURR.json"
    ln "$OUTPUTDIR/edge_types_graph$CURR.json" "edge_types_graph$CURR.json"

    zip "$OUTPUTDIR/graph$CURR.zip" \
                      "graph$CURR.json" \
                      "node_features_graph$CURR.json" \
                      "edge_features_graph$CURR.json" \
                      "node_types_graph$CURR.json" \
                      "edge_types_graph$CURR.json"
    mkdir $OUTPUTDIR/graph$CURR
    mv "$OUTPUTDIR/graph_stats$CURR.txt" $OUTPUTDIR/graph$CURR
    mv "$OUTPUTDIR/graph$CURR.zip" $OUTPUTDIR/graph$CURR
    mv "$OUTPUTDIR/graph$CURR.gpickle" $OUTPUTDIR/graph$CURR
    rm "$OUTPUTDIR/graph$CURR.json"
    rm "$OUTPUTDIR/node_features_graph$CURR.json"
    rm "$OUTPUTDIR/edge_features_graph$CURR.json"
    rm "$OUTPUTDIR/node_types_graph$CURR.json"
    rm "$OUTPUTDIR/edge_types_graph$CURR.json"
    rm "graph$CURR.json"
    rm "node_features_graph$CURR.json"
    rm "edge_features_graph$CURR.json"
    rm "node_types_graph$CURR.json"
    rm "edge_types_graph$CURR.json"
    CURR=$((CURR + 1))
done
