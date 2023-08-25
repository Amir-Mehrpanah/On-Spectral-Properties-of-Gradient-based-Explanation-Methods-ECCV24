import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_shape",
    nargs=4,
    type=int,
    default=(1, 3, 224, 224),
)

args = parser.parse_args()
args.input_shape = tuple(args.input_shape)
print(args.input_shape)
print("type(args.input_shape)", type(args.input_shape))
print("args.input_shape[0]", args.input_shape[0])
