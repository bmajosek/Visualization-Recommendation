import pandas as pd
from llm import LLMInterface
from draco_interface import VisualizeRecommendation

class RecommendationPipeline:
    def __init__(self, df: pd.DataFrame, model_name: str, huggingfacehub_api_token: str):
        """
        Initializes the recommendation pipeline.
        """
        self.df = df
        self.llm_interface = LLMInterface(model_name, huggingfacehub_api_token)
        self.visualization = VisualizeRecommendation(self.llm_interface)

    def run(self, df : pd.DataFrame):
        """
        Visualize two columns based on the Dataframe.
        """
        columns = self.llm_interface.select_columns(df)
        
        self.visualization.recommend_chart(df, columns, True)    
        
    def evaluate(self, df : pd.DataFrame):
        """
        Evaluate all combinations.
        """
        best_pair, best_cost = self.visualization.evaluate_all_column_pairs(df)
        return best_pair, best_cost
