from langchain.prompts import PromptTemplate

RECOMMENDATION_PROMPT = PromptTemplate(
    input_variables=["columns"],
    template="""You are an assistant that MUST return strictly valid JSON nothing else.
Columns: {columns}
Returns JSON describing which columns should be selected to visualize. Return only JSON.

Output Example:
{{"first_column": "name", "second_column": "age"}}
"""
)

EVALUATION_PROMPT = PromptTemplate(
    input_variables=["columns", "chart_spec"],
    template="""You are an expert data analyst that MUST return strictly valid JSON nothing else. Given the data columns: {columns}, evaluate the following Vega-Lite chart specification for its effectiveness in visualizing the data. Provide a score between 0 (poor) to 10 (excellent).

Chart Specification:
{chart_spec}
Return only JSON.
Output Example:
{{"score": <score>}}
"""
)