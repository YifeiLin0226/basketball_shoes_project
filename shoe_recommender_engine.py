import pandas as pd
from typing import List, Dict, Union
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.prompts import PromptTemplate
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool

from shoe_constants import COLUMN_DESCRIPTIONS

class ShoeRecommenderEngine:
    def __init__(self, file_path: str, kwargs: Dict):
        self.df = pd.read_csv(file_path)
        self.columns = COLUMN_DESCRIPTIONS
        # self.llm = Ollama(**kwargs)
        self.llm = OpenAI(**kwargs)

    def _get_column_rank_instructions(self, query: str) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        prompt_template = PromptTemplate(
            template="""
You are a shoe recommendation assistant.

Given a user query and the following column names from a shoe specification table:
{columns}

Determine:
1. Which column(s) should be **ranked** (sorted), and in what direction (asc/desc).
2. Which column(s) should be **filtered**, and the filter condition (e.g., < 150). **Only** filter columns if the query explicitly mentions them, you should not implicitly filter but sort.

Return a valid Python dictionary like this:
{{
  "ranking": [
    {{"column": "Weight oz/g", "sort": "asc"}},
    {{"column": "Cushioning Score", "sort": "desc"}}
  ],
  "filters": [
    {{"column": "Price", "op": "<", "value": 150}}
  ]
}}

and an explanation of your reasoning.
User query: "{query}"
"""
        )

        prompt = prompt_template.format(columns=self.columns, query=query)

        response = self.llm.chat(
            messages=[ChatMessage(role="user", content=prompt)]
        )

        output = response.message.content.strip()

        # separate the dictionary and explanation
        try:
            dict_start = output.index("{")
            dict_end = output.rindex("}") + 1
            dict_str = output[dict_start:dict_end]
            dict_output = eval(dict_str)  # Use eval carefully, ensure the input is trusted
            explanation = output[dict_end:].strip()
        except (ValueError, SyntaxError):
            dict_output = {}
            explanation = output.strip()

        return {"ranking": dict_output.get("ranking", []), "filters": dict_output.get("filters", []), "explanation": explanation}

    def _apply_filters(self, filters: List[Dict[str, Union[str, float]]], df: pd.DataFrame) -> pd.DataFrame:
        for f in filters:
            col, op, val = f["column"], f["op"], f["value"]
            if col not in df.columns:
                continue
            try:
                if op == "<":
                    df = df[df[col] < val]
                elif op == ">":
                    df = df[df[col] > val]
                elif op == "<=":
                    df = df[df[col] <= val]
                elif op == ">=":
                    df = df[df[col] >= val]
                elif op == "==":
                    df = df[df[col] == val]
            except Exception:
                continue
        return df

    async def recommend(self, ctx: Context, query: str, num_shoes : int = 5) -> str:
        """
        Recommends basketball shoes based on preferences like fit, weight, cushioning, or other performance aspects. Do NOT use this when specific shoe names are given.
        Args:
            ctx (Context): The context object for handling events and responses.
            query (str): User's query describing preferences and constraints.
            num_shoes (int): Number of shoes to recommend.
        Returns:
            str: A string containing the recommended shoes and explanation
        """
        
        decision = self._get_column_rank_instructions(query)
        ranking_columns = decision.get("ranking", [])
        filters = decision.get("filters", [])

        filtered_df = self._apply_filters(filters, self.df.copy())

        sorted_df = filtered_df
        for instr in reversed(ranking_columns):
            col = instr["column"]
            ascending = instr["sort"].lower() == "asc"
            if col in sorted_df.columns:
                sorted_df = sorted_df.sort_values(col, ascending=ascending)

        shoes_specs = sorted_df.head(num_shoes).to_dict(orient='records')
        
        state = await ctx.get("state", {})
        recommendations = state.get("recommendations", [])
        recommendations.extend([shoe['NAME'] for shoe in shoes_specs])
        await ctx.set("state", {"recommendations": recommendations})
        return str({"shoes_specs": shoes_specs, "explanation": decision["explanation"]})
    
    def as_tool(self):

        return FunctionTool.from_defaults(
            fn=self.recommend,
            name="shoe_recommender"
        )