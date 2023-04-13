DESC = """
Read a set of archives containing JSON/parquet files, with each archive 
corresponding to a cell. Aggregate data from these files into a table 
that can be used for an ML task.
"""
import json
import argparse
from utils import generate_ml_table


def argument_parser(desc=''):
    parser=argparse.ArgumentParser(description=desc, 
            formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-c', '--cell-files', 
            nargs='*', 
            required=True,
            help='Cell parque/JSON files')

    # Output
    parser.add_argument("-o", "--output",
            help="CSV file to output DF to",
            default='out')

    parser.add_argument("--input-format",
            choices=['json', 'parquet'],
            help="whether input is a JSON file or a parquet file",
            default='parquet')

    parser.add_argument("--out-degree-bin-size",
            nargs='*',
            help="The binning coarseness of out-degree binning",
            type=int,
            default=[2])

    parser.add_argument("--epicurve-bin-size",
            nargs='*',
            help="The binning coarseness of the epicurve binning",
            type=int,
            default=[5])

    parser.add_argument("--boundary-out-degree-bin-size",
            nargs='*',
            help="The binning coarseness of the boundary out-degree",
            type=int,
            default=[5])

    parser.add_argument("--extra-features",
            help="Extra features to be added to the output data set",
            type=json.loads)
    
    
    return parser

def main(args):
    input_format = args.input_format
    output_df_path = args.output
    input_files = args.cell_files
    extra_features = args.extra_features
    out_degree_bin_size = [int(i) for i in args.out_degree_bin_size]
    epicurve_bin_size = [int (i) for i in args.epicurve_bin_size]
    boundary_out_degree_bin_size = [int(i) for i in args.boundary_out_degree_bin_size]
    merged_table = generate_ml_table(input_format, input_files, extra_features, out_degree_bin_size, epicurve_bin_size, boundary_out_degree_bin_size)
    merged_table.to_csv(output_df_path, index=None)

if __name__=='__main__':
    args = argument_parser(DESC).parse_args()
    main(args)