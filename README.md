🧠 AI + SQL Workflow Prototype

Technologies: Python, LangChain, OpenAI GPT, PostgreSQL, SQLAlchemy

This prototype demonstrates an AI-powered SQL Agent that converts natural language questions into executable SQL queries using LangChain and PostgreSQL.
It supports schema grounding, validation, caching, and basic visualization.

🔹 Key Features

Natural Language → SQL using LangChain and ChatOpenAI

Schema awareness: uses extracted schema cards for contextual grounding

Safe execution: validates and prevents destructive SQL (non-SELECT)

Result summarization: converts raw query output into human-readable insights

Caching & Validation: avoids re-running repeated queries

Visualization hooks: simple Matplotlib plotting for top results

🏗️ Architecture Overview

Database layer – SQLAlchemy engine connects to a local PostgreSQL instance.

Schema extraction – a Python script generates Markdown/JSON schema summaries.

Direct NL→SQL – deterministic model (no intermediate reasoning chain).

LangChain Agent – reasoning-style SQL agent with controlled routing and safety rules.

Validation & Caching – file-based caching and dataframe validation.

Visualization – quick, compliant Matplotlib charts for insight presentation.
