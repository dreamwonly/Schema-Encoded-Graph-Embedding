# import tlsh

def one_hot_encode(idx, len):
    ret = [0] * len
    ret[idx] = 1
    return ret

def reverse_dict(dictionary):
    revdict = {}
    for key in dictionary:
        revdict[dictionary[key]] = key
    return revdict

def hash_features(features):
    concat_feat = ""
    for feature in features:
        pieces = feature.split(":")
        if len(pieces) == 1:
            concat_feat = concat_feat + pieces[0]
        else:
            concat_feat = concat_feat + pieces[1]
    return tlsh.hash(str.encode(concat_feat))

def sanitize_features(features):
    if features is None:
        return features
    clean_features = {}
    if "cf:offset" in features.keys():
        features.pop("cf:offset")
    for feature in features.keys():
        clean_features[feature.replace("cf:", "").replace("prov:","")] = str(features[feature]).replace("cf:","").replace("prov:","").replace(":","-")
    return clean_features
