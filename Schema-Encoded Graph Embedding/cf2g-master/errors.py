##################
# error messages #
##################

def input_error_alert():
    print("JSON input not loaded - did you forget to call the loader first?")

def feature_length_alert():
    print("Inconsistent feature vector lengths detected in the same node/edge type.")

def format_unrecognized():
    print("ERROR: format unrecognized. Supported: w3cprov, spade. Run -h for help.")

def input_required():
    print("ERROR: input path required. Run -h for help.")

def relation_error():
    print("ERROR: relation for W3C PROV JSON unrecognized.")
