from collections import namedtuple
import numpy as np

Label = namedtuple('Label', [

    'name',  # The identifier of this label

    'id',  # An integer ID that is associated with this label.

    'trainId',

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',

    'color',  # The color of this label
])

# --------------------------------------------------------------------------------
# A list of all labels
# --------------------------------------------------------------------------------

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('unlabeled', 0, 0, 'void', 0, False, False, (68, 4, 81)),
    Label('sclera', 1, 1, 'void', 0, False, False, (51, 101, 139)),
    Label('iris', 2, 2, 'void', 0, False, False, (55, 182, 121)),
    Label('pupil', 3, 3, 'void', 0, False, False, (253, 230, 67)),
]

# --------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
# --------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label = {label.trainId: label.name for label in labels}
# id to label object
# id2label        = { label.id   : label for label in labels           }
# print(id2label.items())
id2label = {label.id: label for label in labels[:2]}
# print(id2label)

# trainId to label object
trainId2label = {label.trainId: label for label in labels}

label2trainId = {v: k for k, v in trainId2label.items()}
