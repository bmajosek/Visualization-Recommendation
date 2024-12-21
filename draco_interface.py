import pandas as pd
from draco import Draco, answer_set_to_dict
from draco.renderer import AltairRenderer
from draco.fact_utils import dict_to_facts
from IPython.display import display
from draco.run import run_clingo
from llm import LLMInterface

class VisualizeRecommendation:
    def __init__(self, llm : LLMInterface):
        self.draco = Draco()
        self.renderer = AltairRenderer()
        self.llm = llm

    def recommend_chart(self, df: pd.DataFrame, columns: dict, display_charts : bool):
        """
        Generates a recommended chart and evaluates its effectiveness using an LLM.
        """
        float_cols = df.select_dtypes(include=['float64', 'float32', 'float']).columns
        for col in float_cols:
            df[col] = df[col].astype(str).apply(lambda x: f"val{x.replace('.', '_')}")
            
        schema = {
            "number_rows": len(df),
            "field": [
                {"name": col, "type": "number" if df[col].dtype in ['int64', 'float64'] else "string"}
                for col in df.columns
            ]
        }
        
        vega_lite_spec = {
            "view": [
                {
                    "coordinates": "cartesian",
                    "mark": [
                        {
                            "type": "point",
                            "encoding": [
                                {"channel": "x", "field": columns["first_column"]},
                                {"channel": "y", "field": columns["second_column"]}
                            ]
                        }
                    ]
                }
            ]
        }

        spec = {**schema, **vega_lite_spec}
        asp_facts = dict_to_facts(spec)
        violations = self.draco.get_violations(asp_facts)
        model = next(run_clingo(asp_facts))
        answer_set = model.answer_set
        spec = answer_set_to_dict(answer_set)
        chart = self.renderer.render(spec, data=df)
        if display_charts:
            display(chart)
        
        cost = self.llm.evaluate_chart(columns, spec)
        return cost, violations, model.cost
    

    def evaluate_all_column_pairs(self, df: pd.DataFrame):
        """
        Evaluates all possible column pairs in the DataFrame and identifies the best pair for visualization.
        """
        columns = df.columns
        best_evaluation_score = float('-inf')
        best_model_cost = float('inf')
        best_violations = float('inf')

        best_evaluation_pair = None
        best_cost_pair = None

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                evaluation, violations, model_cost = self.recommend_chart(df, {"first_column": columns[i], "second_column": columns[j]}, False)
                print(f"Columns: {columns[i]}, {columns[j]} - Evaluation Score: {evaluation['score']} - Model Cost: {model_cost} - Violations: {violations}")

                if evaluation['score'] > best_evaluation_score:
                    best_evaluation_score = evaluation['score']
                    best_evaluation_pair = (columns[i], columns[j])

                if model_cost and (sum(model_cost) < best_model_cost or (sum(model_cost) == best_model_cost and len(violations) < best_violations)):
                    best_model_cost = sum(model_cost)
                    best_violations = len(violations)
                    best_cost_pair = (columns[i], columns[j])

        print(f"Best column pair by evaluation score: {best_evaluation_pair} with score: {best_evaluation_score}")
        print(f"Best column pair by model cost and violations: {best_cost_pair} with cost: {best_model_cost} and violations: {best_violations}")
        
        return best_evaluation_pair, best_evaluation_score, best_cost_pair, best_model_cost, best_violations

