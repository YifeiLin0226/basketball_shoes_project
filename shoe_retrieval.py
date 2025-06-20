import pandas as pd
from llama_index.readers.file import PagedCSVReader
from llama_index.core import VectorStoreIndex, load_index_from_storage, StorageContext
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context   

from typing import List
import os

from shoe_constants import COLUMN_DESCRIPTIONS

class ShoeRetrieval:
    def __init__(self, file_path: str):

        self.df = pd.read_csv(file_path)
        self.median_values = self.df.median(numeric_only=True)

        if not os.path.exists("./storage"):
            documents = PagedCSVReader().load_data(file_path)
            self.index = VectorStoreIndex.from_documents(documents)
            self.index.storage_context.persist("./storage")
        else:
            storage_context = StorageContext.from_defaults(persist_dir="./storage")
            self.index = load_index_from_storage(storage_context)
    
    def retrieve(self, names: List[str]):
        """
        Retrieve basketball shoes based on user query.
        Args:
            name (list[str]): List of basketball shoe names to retrieve.
        """
        
        response = []
        retriever = self.index.as_retriever(similarity_top_k=1)
        for name in names:
            results = retriever.retrieve(name)
            response.extend([r.node.get_content() for r in results])
        return '\n'.join(response)
    
    def retrieve_with_median(self, names: List[str]):
        """
        Given the name of one or more basketball shoes, returns their specs for direct comparison. Use this only when specific shoe name(s) are mentioned.
        Args:
            name (List[str]): List of basketball shoe names to retrieve.
        """
        response = self.retrieve(names)
        response_with_stats = {
            "shoes_specs": response,
            "global_median_values": self.median_values.to_dict(),
            "column_descriptions": COLUMN_DESCRIPTIONS
        }
        
        return str(response_with_stats)
        
    
    def as_tool(self):
        return FunctionTool.from_defaults(
            fn=self.retrieve_with_median,
            name="shoe_retriever",
        )