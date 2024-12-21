import pandas as pd
import argparse
from pipeline import RecommendationPipeline
from vega_datasets import data

def parse_args():
    """
    Parses command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(description='Visualization recommendation.')
    parser.add_argument('model_name', type=str, help='Model name from HuggingFace')
    parser.add_argument('api_token', type=str, help='Hugging Face API token')
    return parser.parse_args()

def main(args):
    """
    Main function to visualization recommendation.
    """
    df = data.iris()
    pipeline = RecommendationPipeline(
        df, args.model_name, args.api_token
    )
    pipeline.run(df)

if __name__ == "__main__":
    args = parse_args()
    main(args)
