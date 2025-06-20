# 🏀 Basketball Shoes Multi-Agent Workflow

![Demo](demo.gif)

🔗 **[Read the Medium Post](https://medium.com/@bobbylin0226/building-a-basketball-shoe-ai-chatbot-with-llamaindex-and-multi-agent-workflows-3c391ba0e646)**


## 🤖 Agents in Action

This project uses an LLM-powered multi-agent architecture to help sneakerheads and ballers alike:

1. **Shoe Recommender Agent** — suggests basketball shoes based on your play style, preferences, and needs.  
2. **Shoe Retriever Agent** — fetches and analyzes detailed specs of specific models on request.

---

## ⚙️ Project Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.  
To install the dependencies, run:

```bash
poetry install
```
## 📊 Dataset
The dataset is provided by Foot Doctor Zach for [purchase](https://www.footdoctorzach.com/6c0119d4-ae79-4ec5-823d-509d43a39ed5), please put it under `data/` 

`data_process.ipynb` processes the raw data by fixing numeric types and creating boolean values. This helps the LLM understand and apply filters.

## 💬 Launch the Chatbot
```
chainlit run app.py -w
```