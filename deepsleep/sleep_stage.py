# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

NUM_CLASSES = 5  # exclude UNKNOWN

class_dict_5 = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}

class_dict_4 = {
    0: "W",
    1: "Light",
    2: "Deep",
    3: "REM"
}

class_dict_3 = {
    0: "QS",
    1: "AS",
    2: "W"
}

class_dict_2 = {
    1: "W",
    0: "S"
}

EPOCH_SEC_LEN = 30  # seconds
SAMPLING_RATE = 128

def print_n_samples_each_class_5(labels): 
    import numpy as np
    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        print ("{}: {}".format(class_dict_5[c], n_samples))
        
def print_n_samples_each_class_4(labels):
    import numpy as np
    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        print ("{}: {}".format(class_dict_4[c], n_samples))

def print_n_samples_each_class_3(labels):
    import numpy as np
    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        print ("{}: {}".format(class_dict_3[c], n_samples))

def print_n_samples_each_class_2(labels):
    import numpy as np
    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        print ("{}: {}".format(class_dict_2[c], n_samples))
