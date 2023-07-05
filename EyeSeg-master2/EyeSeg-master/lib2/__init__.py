from .helper_pytorch2 import flatten, make_one_hot, pred2ones, \
                            get_predictions, per_class_mIoU, pp_class_miou, \
                            labelid_to_color, encode_test, torch_validate, \
                            pred2ones, one_hot2dist
from .command_line import command_line_args

args = command_line_args().parse_args()