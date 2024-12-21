import re
import pandas as pd
import json
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from prompts import RECOMMENDATION_PROMPT, EVALUATION_PROMPT

class LLMInterface:
    def __init__(self, model_name: str, huggingfacehub_api_token: str):
        """
        Initializes the LLMTransformer with a Hugging Face model and API token.
        """
        self.llm = HuggingFaceHub(
            repo_id=model_name,
            huggingfacehub_api_token=huggingfacehub_api_token,
            model_kwargs={"max_length": 500}
        )
        self.chain = LLMChain(llm=self.llm, prompt=RECOMMENDATION_PROMPT)
        self.evaluation_chain = LLMChain(llm=self.llm, prompt=EVALUATION_PROMPT)

    def select_columns(self, df : pd.DataFrame):
        """
        Pick two columns based on a Dataframe.
        """
        columns = ", ".join(df.columns)
        llm_response = self.chain.run(columns=columns)
        match = re.findall(r"{\s*.*?\s*}", llm_response, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON found in LLM response.")
        json_array_str = match[-1].strip()
        return json.loads(json_array_str)
    
    def evaluate_chart(self, columns: str, chart_spec: str):
        """
        Evaluate picking two columns based on a chart_spec using LLM.
        """
        llm_response = self.evaluation_chain.run(columns=columns, chart_spec=chart_spec)
        match = re.findall(r"{\s*.*?\s*}", llm_response, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON found in LLM response.")
        
        json_array_str = match[-1].strip()
        return json.loads(json_array_str)