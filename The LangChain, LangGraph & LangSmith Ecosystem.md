# AI Agents Mastery: The LangChain, LangGraph & LangSmith Ecosystem

---

## Part 1: The Paradigm Shift — From Chains to Agents

### Why LangGraph Exists

**In plain English:** Vanilla LangChain chains are Directed Acyclic Graphs (DAGs) — data flows one direction, step A always leads to step B, and there's no way to loop back and try again. Agents need loops: "call a tool, look at the result, decide whether to call another tool, or answer." A DAG can't express "go back to step 2 if the condition fails." LangGraph models your application as a **graph with cycles**, not a pipeline, which is the actual shape of agentic reasoning (ReAct, reflection, self-correction).

Key structural differences:

| Aspect | LCEL Chains (DAG) | LangGraph (Cyclic State Machine) |
|---|---|---|
| Control flow | Linear, one-directional | Cycles, branches, loops |
| State | Passed implicitly between runnables | Explicit, shared `State` object |
| Conditional routing | Limited (`RunnableBranch`) | First-class (`add_conditional_edges`) |
| Persistence | None built-in | Checkpointers (memory survives crashes) |
| Best for | RAG pipelines, simple transforms | Agents, multi-step reasoning, HITL |

### The Core Loop: `StateGraph` Architecture

A `StateGraph` has three primitives:
- **State**: a shared schema (TypedDict or Pydantic model) every node reads from and writes to.
- **Nodes**: plain Python functions (or Runnables) that take the state and return a partial update.
- **Edges**: connections between nodes, either fixed or conditional (a function decides where to go next).

### Code Example: Your First `StateGraph`

```python
# pip install langgraph>=0.2.0
from typing import TypedDict
from langgraph.graph import StateGraph, END

class CounterState(TypedDict):
    count: int
    max_count: int

def increment(state: CounterState) -> dict:
    # Nodes return a *partial* update dict, not the full state
    return {"count": state["count"] + 1}

def should_continue(state: CounterState) -> str:
    # Conditional edge function: returns the NAME of the next node
    if state["count"] < state["max_count"]:
        return "loop"   # <-- route back to the increment node
    return "done"       # <-- route to END

builder = StateGraph(CounterState)
builder.add_node("increment", increment)
builder.set_entry_point("increment")

# Conditional edge: after "increment", call should_continue to decide next hop
builder.add_conditional_edges(
    "increment",
    should_continue,
    {"loop": "increment", "done": END}  # <-- map return values to nodes
)

graph = builder.compile()

result = graph.invoke({"count": 0, "max_count": 5})
print(result)  # {'count': 5, 'max_count': 5}
```

*Gotcha: `set_entry_point` and `add_conditional_edges` mutate the builder, not a copy — always call `.compile()` last, and never reuse a compiled graph object as if it were still editable.*

---

## Part 2: LangChain Expression Language (LCEL) — The Glue

### The `|` Operator

**In plain English:** LCEL overloads Python's `|` (pipe) operator so any two `Runnable` objects can be composed like Unix pipes: output of the left becomes input of the right. This is how you build the individual "tool" or "retrieval" pieces that a LangGraph node will call.

### `RunnableParallel` & `RunnablePassthrough`

- `RunnableParallel` runs multiple Runnables concurrently against the same input and merges their outputs into a dict.
- `RunnablePassthrough` forwards the original input unchanged alongside transformed branches — essential when you need both the raw question and retrieved context downstream.

### Code Example: LCEL Retrieval Pipeline Feeding an Agent

```python
# pip install langchain langchain-openai langchain-community faiss-cpu
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Build a toy vector store
docs = ["LangGraph uses a StateGraph for cyclic agent workflows.",
        "Checkpointers persist state across threads and restarts."]
vectorstore = FAISS.from_texts(docs, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_template(
    "Answer using only this context:\n{context}\n\nQuestion: {question}"
)
llm = ChatOpenAI(model="gpt-4o-mini")

def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)

# RunnableParallel fans the input into two branches: context + passthrough question
rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,      # <-- retrieve then format
        question=RunnablePassthrough(),        # <-- forward original input untouched
    )
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What does a checkpointer do?")
print(answer)
```

*Gotcha: `RunnableParallel` branches execute concurrently via threads under the hood — don't share mutable Python objects across branches, since order of completion isn't guaranteed.*

This `rag_chain` Runnable can be dropped directly into a LangGraph node as-is (`rag_chain.invoke(...)` inside a node function) — LCEL and LangGraph compose, they don't compete.

---

## Part 3: LangGraph Deep Dive (The Engine)

### State Management: TypedDict vs. Pydantic

| | `TypedDict` | Pydantic `BaseModel` |
|---|---|---|
| Runtime validation | None | Yes (raises on bad types) |
| Performance | Faster (no validation overhead) | Slightly slower |
| Default values | Awkward (need `total=False`) | Native `Field(default=...)` |
| Recommended for | Simple internal state | State exposed to external callers / APIs |

### Nodes: `call_model`, `call_tool`, and Custom Logic

### Edges & Conditional Edges

### Checkpointers (Persistence)

| Checkpointer | Use Case | Persistence | Notes |
|---|---|---|---|
| `MemorySaver` | Local dev, tests | In-process only, lost on restart | Fastest, zero setup |
| `SqliteSaver` | Single-instance prod, small apps | Disk file | Simple, no extra infra |
| `PostgresSaver` | Multi-instance production | Postgres DB | Supports horizontal scaling |
| `RedisSaver` | High-throughput, low-latency prod | Redis | Fast reads/writes, needs TTL policy |

### Human-in-the-Loop (HITL)

`interrupt_before=["node_name"]` (or `interrupt_after`) pauses the graph right before/after a node runs, returning control to your application so a human can approve, edit state, or reject before execution resumes with `graph.invoke(None, config)`.

### Code Example: ReAct Agent with Persistent Memory

```python
# pip install langgraph langchain-openai langchain-core
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage

@tool
def get_weather(city: str) -> str:
    """Return the current weather for a given city."""
    return f"It's sunny in {city}."

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools([get_weather])

class AgentState(TypedDict):
    # add_messages reducer appends new messages instead of overwriting the list
    messages: Annotated[list[BaseMessage], add_messages]

def call_model(state: AgentState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # <-- reducer merges this into existing list

def call_tool(state: AgentState) -> dict:
    last_msg = state["messages"][-1]
    tool_call = last_msg.tool_calls[0]
    result = get_weather.invoke(tool_call["args"])
    from langchain_core.messages import ToolMessage
    return {"messages": [ToolMessage(content=result, tool_call_id=tool_call["id"])]}

def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    return "tools" if getattr(last_msg, "tool_calls", None) else "end"

builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", call_tool)
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")  # <-- loop back after tool execution

# Checkpointer persists state per-thread to a local SQLite file — survives restarts
with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "user-42"}}
    graph.invoke({"messages": [("user", "What's the weather in Tokyo?")]}, config)

    # Later — even after a process restart — resume the same thread:
    result = graph.invoke({"messages": [("user", "And in Paris?")]}, config)
    print(result["messages"][-1].content)
```

*Gotcha: Without a reducer (`Annotated[list, add_messages]`), returning `{"messages": [...]}` from a node **replaces** the whole list rather than appending — you'd lose conversation history every turn.*

---

## Part 4: Advanced LangGraph Patterns ("The Super Stuff")

### Subgraphs

A compiled graph is itself a valid node. This enables hierarchical composition: a "Researcher" subgraph, a "Coder" subgraph, and a "Reviewer" subgraph can each be built, tested, and compiled independently, then wired together as nodes in a parent `Supervisor` graph.

### Streaming

- `stream_mode="updates"` yields only the delta returned by each node as it finishes.
- `stream_mode="custom"` lets a node emit arbitrary events (e.g., token-by-token text) via `get_stream_writer()` inside the node, useful for piping partial LLM output to a frontend before the node completes.

### Time Travel / Replay

Every checkpoint has a `checkpoint_id`. Calling `graph.get_state_history(config)` returns every past state; passing an earlier `checkpoint_id` back into `config["configurable"]` and re-invoking rewinds execution to that point — useful for debugging or letting a user "undo" an agent's last action.

### Multi-Agent Collaboration: The Supervisor Pattern

A `Supervisor` node is itself an LLM call whose only job is routing: given the conversation so far, decide which specialized subgraph should act next (Researcher, Coder, Reviewer), or finish.

### Code Example: Supervisor with 3 Subgraphs, Streaming, and Error Handling

```python
# pip install langgraph langchain-openai
from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage

class TeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str

llm = ChatOpenAI(model="gpt-4o-mini")

def build_worker(role: str):
    """Factory that builds a minimal one-node subgraph for a given role."""
    def worker_node(state: TeamState) -> dict:
        try:
            response = llm.invoke(
                [HumanMessage(content=f"As the {role}, respond to: {state['messages'][-1].content}")]
            )
            return {"messages": [response]}
        except Exception as e:  # <-- error resiliency: never let a worker crash the graph
            return {"messages": [HumanMessage(content=f"[{role} failed: {e}]")]}

    sub_builder = StateGraph(TeamState)
    sub_builder.add_node("work", worker_node)
    sub_builder.set_entry_point("work")
    sub_builder.add_edge("work", END)
    return sub_builder.compile()

researcher_graph = build_worker("Researcher")
coder_graph = build_worker("Coder")
reviewer_graph = build_worker("Reviewer")

def supervisor(state: TeamState) -> Command[Literal["researcher", "coder", "reviewer", "__end__"]]:
    # Supervisor LLM call decides routing; kept deterministic here for brevity
    decision = llm.invoke(
        [HumanMessage(content=f"Given: {state['messages'][-1].content}\n"
                               f"Reply with exactly one word: researcher, coder, reviewer, or done.")]
    ).content.strip().lower()
    target = decision if decision in ("researcher", "coder", "reviewer") else "__end__"
    return Command(goto=target)  # <-- Command lets a node both update state and route

builder = StateGraph(TeamState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher_graph)  # <-- subgraph used directly as a node
builder.add_node("coder", coder_graph)
builder.add_node("reviewer", reviewer_graph)

builder.set_entry_point("supervisor")
for worker in ("researcher", "coder", "reviewer"):
    builder.add_edge(worker, "supervisor")  # <-- always report back to supervisor

graph = builder.compile()

# Streaming: print each node's incremental update as it happens
for chunk in graph.stream(
    {"messages": [HumanMessage(content="Find recent LangGraph release notes")], "next_agent": ""},
    stream_mode="updates",
):
    print(chunk)  # <-- {'researcher': {'messages': [...]}} etc.
```

*Gotcha: Returning `Command(goto=...)` from a node bypasses `add_conditional_edges` entirely — mixing both routing styles in the same graph gets confusing fast; pick one per graph.*

---

## Part 5: LangSmith — Observability & Evaluation Framework

### Tracing

Every LCEL/LangGraph invocation, when `LANGCHAIN_TRACING_V2=true` is set, automatically logs a **Run** tree to LangSmith: the parent chain/graph run, and every child run (each LLM call, each tool call), with latency, token counts, and cost attached. No code changes needed beyond environment variables.

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls__...
export LANGCHAIN_PROJECT=my-agent-prod
```

### Datasets

A LangSmith Dataset is a versioned collection of `(input, expected_output)` examples — your regression suite. Curate these from real production traces you've labeled as good/bad.

### Evaluators

| Type | What it does | When to use |
|---|---|---|
| Heuristic | Regex, exact-match, JSON schema check | Deterministic, structured outputs |
| LLM-as-a-Judge | A strong LLM scores relevance/correctness/tone | Open-ended, subjective quality |
| Pairwise | Compares two responses, picks a winner | A/B testing prompt or model changes |
| Custom Scorer | Arbitrary Python function (e.g., citation check) | Domain-specific correctness rules |

### The `evaluate` Function & Feedback

### Code Example: Dataset → Custom Judge → Evaluation Run → Report

```python
# pip install langsmith>=0.1.0 langchain-openai
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI

client = Client()

# 1. Create a dataset of golden examples
dataset = client.create_dataset(dataset_name="agent-qa-golden-v1")
examples = [
    {"question": "What checkpointer should I use in production?", "expected": "Postgres or Redis"},
    {"question": "What operator composes LCEL runnables?", "expected": "the pipe operator |"},
]
for ex in examples:
    client.create_example(
        inputs={"question": ex["question"]},
        outputs={"expected": ex["expected"]},
        dataset_id=dataset.id,
    )

# 2. Define a custom LLM-as-Judge evaluator
judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def llm_judge(run, example) -> dict:
    prediction = run.outputs.get("answer", "")
    expected = example.outputs.get("expected", "")
    verdict = judge_llm.invoke(
        f"Question: {example.inputs['question']}\n"
        f"Expected concept: {expected}\n"
        f"Agent answer: {prediction}\n"
        f"Does the answer correctly convey the expected concept? Reply YES or NO only."
    ).content.strip().upper()
    return {"key": "correctness", "score": 1 if verdict.startswith("YES") else 0}

# 3. The target function is whatever you're evaluating — your compiled LangGraph agent
def target(inputs: dict) -> dict:
    # graph is your compiled StateGraph from Part 3
    result = graph.invoke({"messages": [("user", inputs["question"])]})
    return {"answer": result["messages"][-1].content}

# 4. Run the evaluation and get a summary report
results = evaluate(
    target,
    data="agent-qa-golden-v1",
    evaluators=[llm_judge],
    experiment_prefix="agent-v1-eval",
)

# Print pass/fail summary
scores = [r["evaluation_results"]["results"][0].score for r in results]
pass_rate = sum(scores) / len(scores) * 100
print(f"Pass rate: {pass_rate:.1f}% ({sum(scores)}/{len(scores)})")
```

*Gotcha: `evaluate()` runs your `target` function against **every** example concurrently by default — if your agent has side effects (writes to a real DB, sends real emails) you must mock those out before running evals, or you'll spam production.*

---

## Part 6: Tools & The Model Context Protocol (MCP)

### Defining Tools: `@tool` and Docstrings

**In plain English:** The LLM never sees your Python code — it only sees the tool's name, its docstring, and its argument schema (inferred from type hints). A vague docstring produces vague tool calls. Treat the docstring as the entire API contract you're handing the model.

### Error Resiliency

### MCP Integration

LangGraph agents can attach to any MCP-compliant server via `langchain-mcp-adapters`, which converts each MCP tool into a native LangChain `Tool` object automatically — the agent calls them exactly like local `@tool`-decorated functions.

### Code Example: Robust Tool Set with Graceful Error Reporting

```python
# pip install langchain-core
from langchain_core.tools import tool
import json

@tool
def web_search(query: str) -> str:
    """Search the web for a query and return a short summary of the top result.
    Use this for questions about current events or facts not in your training data."""
    try:
        # placeholder for a real search call
        if not query.strip():
            raise ValueError("query cannot be empty")
        return f"Top result for '{query}': [summary would go here]"
    except Exception as e:
        # Report failure back to the LLM as text, not a raised exception —
        # a raised exception crashes the graph; a string lets the agent retry or rephrase.
        return f"TOOL_ERROR: web_search failed — {e}. Try rephrasing the query."

@tool
def db_lookup(table: str, record_id: str) -> str:
    """Look up a record by ID in a given database table. Returns JSON or an error string."""
    valid_tables = {"users", "orders"}
    if table not in valid_tables:
        return f"TOOL_ERROR: unknown table '{table}'. Valid tables: {sorted(valid_tables)}"
    try:
        record_id_int = int(record_id)  # <-- malformed args are common; validate explicitly
    except ValueError:
        return f"TOOL_ERROR: record_id must be numeric, got '{record_id}'"
    return json.dumps({"table": table, "id": record_id_int, "status": "found"})

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '2 * (3 + 4)'. No variables or imports."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "TOOL_ERROR: expression contains disallowed characters"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # <-- sandboxed eval
    except Exception as e:
        return f"TOOL_ERROR: could not evaluate expression — {e}"
```

```python
# MCP integration example
# pip install langchain-mcp-adapters
from langchain_mcp_adapters.client import MultiServerMCPClient

async def load_mcp_tools():
    client = MultiServerMCPClient(
        {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()  # <-- returns native LangChain Tool objects
    return tools  # bind these directly: llm.bind_tools(tools)
```

*Gotcha: Tools that raise Python exceptions instead of returning an error string will crash the entire graph run mid-execution — always catch and return `TOOL_ERROR: ...` so the agent can see the failure and recover.*

---

## Part 7: Memory, State, & Persistence in Production

### Short-term vs. Long-term Memory

- **Short-term**: the `messages` list within a single thread — handled by the checkpointer and `thread_id`.
- **Long-term**: facts that persist *across* threads for a given user (preferences, profile facts) — stored separately, keyed by a stable `user_id` passed through `config["configurable"]`.

### State Reducers

A reducer is a function `(current_value, new_value) -> merged_value` attached to a state field via `Annotated[Type, reducer_fn]`. `add_messages` is the built-in reducer for chat history; you can write your own for other accumulating fields (e.g., a running list of tool-call errors).

### Code Example: Cross-Thread User Preference Memory

```python
# pip install langgraph langchain-openai
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore  # <-- long-term, cross-thread key-value store
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOpenAI(model="gpt-4o-mini")

def call_model(state: ChatState, config: dict, *, store) -> dict:
    user_id = config["configurable"]["user_id"]
    # Long-term store is keyed by (namespace, key) — independent of thread_id
    memory = store.get(("preferences", user_id), "style") or {"value": "no preference set"}

    system = SystemMessage(content=f"User preference: {memory['value']}. Follow it strictly.")
    response = llm.invoke([system] + state["messages"])

    # If the user states a new preference, persist it for every future thread
    if "always" in state["messages"][-1].content.lower():
        store.put(("preferences", user_id), "style", {"value": state["messages"][-1].content})

    return {"messages": [response]}

builder = StateGraph(ChatState)
builder.add_node("agent", call_model)
builder.set_entry_point("agent")
builder.add_edge("agent", END)

with SqliteSaver.from_conn_string("threads.sqlite") as checkpointer, \
     SqliteStore.from_conn_string("longterm.sqlite") as store:
    graph = builder.compile(checkpointer=checkpointer, store=store)

    config = {"configurable": {"thread_id": "session-1", "user_id": "alice"}}
    graph.invoke({"messages": [("user", "Always summarize your answers in bullet points.")]}, config)

    # New thread, same user — preference persists because it's in the long-term store, not the thread
    config2 = {"configurable": {"thread_id": "session-2", "user_id": "alice"}}
    result = graph.invoke({"messages": [("user", "Explain how checkpointers work.")]}, config2)
    print(result["messages"][-1].content)
```

*Gotcha: `thread_id` scopes the checkpointer (short-term); `user_id` + the `store` scopes long-term memory. Confusing the two is the #1 cause of "why doesn't my agent remember preferences across sessions" bugs.*

---

## Part 8: Evaluation & Testing Methodology (Expanded)

### Unit Testing Nodes

Test node functions as plain Python functions — no LLM, no graph. Pass a hand-built `state` dict, assert on the returned partial update.

### Integration Testing with Mocked LLMs

Swap the real `ChatOpenAI` for a fake `Runnable` that returns canned responses, then run `graph.invoke()` end-to-end to verify routing logic without spending tokens.

### Continuous Evaluation in CI/CD

### Metrics to Track

| Metric | What it measures |
|---|---|
| Task Success Rate | % of eval examples the agent solves correctly |
| Tool Call Accuracy | % of tool calls with correct name + valid args |
| Latency (p50/p95) | End-to-end response time distribution |
| Cost-per-task | Total token spend / number of tasks |

### Code Example: pytest Suite Gating CI on LangSmith Eval Results

```python
# test_agent_eval.py
# pip install pytest langsmith langchain-core
import pytest
from unittest.mock import MagicMock
from langsmith.evaluation import evaluate

# --- Unit test: isolate a single node, no LLM involved ---
def test_should_continue_routes_to_tools_when_tool_call_present():
    from my_agent.graph import should_continue
    fake_msg = MagicMock(tool_calls=[{"name": "get_weather", "args": {}, "id": "1"}])
    state = {"messages": [fake_msg]}
    assert should_continue(state) == "tools"

def test_should_continue_routes_to_end_when_no_tool_call():
    from my_agent.graph import should_continue
    fake_msg = MagicMock(tool_calls=[])
    state = {"messages": [fake_msg]}
    assert should_continue(state) == "end"

# --- Integration test: full graph, mocked LLM ---
def test_graph_end_to_end_with_mocked_llm(monkeypatch):
    from my_agent.graph import builder
    fake_response = MagicMock(content="mocked answer", tool_calls=[])
    monkeypatch.setattr("my_agent.graph.llm.invoke", lambda *a, **k: fake_response)
    graph = builder.compile()
    result = graph.invoke({"messages": [("user", "hello")]})
    assert result["messages"][-1].content == "mocked answer"

# --- Continuous evaluation gate: fails CI if success rate < 90% ---
def test_langsmith_eval_gate_success_rate():
    from my_agent.graph import graph

    def target(inputs: dict) -> dict:
        result = graph.invoke({"messages": [("user", inputs["question"])]})
        return {"answer": result["messages"][-1].content}

    def correctness(run, example) -> dict:
        score = 1 if example.outputs["expected"].lower() in run.outputs["answer"].lower() else 0
        return {"key": "correctness", "score": score}

    results = evaluate(target, data="agent-qa-golden-v1", evaluators=[correctness])
    scores = [r["evaluation_results"]["results"][0].score for r in results]
    success_rate = sum(scores) / len(scores)

    assert success_rate >= 0.90, f"Success rate {success_rate:.0%} fell below 90% threshold"
```

```yaml
# .github/workflows/eval.yml — runs the gate on every PR
name: Agent Evaluation
on: [pull_request]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest test_agent_eval.py -v
        env:
          LANGCHAIN_API_KEY: ${{ secrets.LANGCHAIN_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

*Gotcha: Mocking `llm.invoke` at the module level (`monkeypatch.setattr("my_agent.graph.llm.invoke", ...)`) only works if the node calls `llm.invoke` via that imported reference — if the node captures `llm` in a closure at import time, patch the closure's target, not a re-imported copy.*

---

## Part 9: Deployment Architecture for Agents

### LangGraph Platform / API

LangGraph Platform packages a compiled graph as a deployable service with built-in persistence, horizontal scaling, and a REST/streaming API — you point it at a Python module exposing a `graph` object and it handles the rest.

### State Management in the Cloud

Swap `SqliteSaver`/`SqliteStore` for `PostgresSaver`/`PostgresStore` the moment you run more than one instance — SQLite is a single file and doesn't coordinate across processes.

### Event Streaming

Expose the graph's `.astream_events()` output over Server-Sent Events (SSE) so a frontend can render the agent's intermediate tool calls and streaming tokens in real time.

### Authentication

### Code Example: FastAPI Wrapper with a `/stream` SSE Endpoint

```python
# pip install fastapi uvicorn sse-starlette langgraph
import json
import os
from fastapi import FastAPI, Depends, HTTPException, Header
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from my_agent.graph import builder
from langgraph.checkpoint.postgres import PostgresSaver

app = FastAPI()

DB_URI = os.environ["DATABASE_URL"]
checkpointer_cm = PostgresSaver.from_conn_string(DB_URI)
checkpointer = checkpointer_cm.__enter__()  # kept open for app lifetime
graph = builder.compile(checkpointer=checkpointer)

API_KEYS = set(os.environ.get("VALID_API_KEYS", "").split(","))

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")  # <-- simple key auth
    return x_api_key

class ChatRequest(BaseModel):
    thread_id: str
    user_id: str
    message: str

@app.post("/stream")
async def stream_agent(req: ChatRequest, api_key: str = Depends(verify_api_key)):
    config = {"configurable": {"thread_id": req.thread_id, "user_id": req.user_id}}

    async def event_generator():
        async for event in graph.astream_events(
            {"messages": [("user", req.message)]}, config, version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    yield {"event": "token", "data": chunk}  # <-- token-by-token to client
            elif event["event"] == "on_chain_end" and event["name"] == "LangGraph":
                yield {"event": "done", "data": json.dumps({"status": "complete"})}

    return EventSourceResponse(event_generator())

@app.on_event("shutdown")
def shutdown():
    checkpointer_cm.__exit__(None, None, None)
```

*Gotcha: `astream_events` emits events for every internal Runnable, including nested LCEL chains inside a node — filter aggressively on `event["event"]` and `event["name"]`, or your SSE stream will flood the client with noise.*

---

## Part 10: The Golden Rules of Agent Engineering

### Structured Outputs

Force reliable JSON with `llm.with_structured_output(YourPydanticModel)` instead of prompting "respond in JSON" and hoping — this uses function-calling under the hood so the model is constrained to a valid schema.

```python
from pydantic import BaseModel

class Verdict(BaseModel):
    approved: bool
    reason: str

structured_llm = llm.with_structured_output(Verdict)
result: Verdict = structured_llm.invoke("Should we approve this refund request?")
print(result.approved, result.reason)
```

### The "Max Iterations" Fallback

Always cap loop-prone graphs with `graph.invoke(input, config, {"recursion_limit": 25})` — without it, a stuck conditional edge (agent keeps calling the same failing tool) runs forever and burns tokens.

### Prompt Injection Defense

Never interpolate raw tool output or user-supplied text directly into a system prompt that carries privileged instructions. Wrap untrusted content in clearly delimited blocks (e.g., `<untrusted_data>...</untrusted_data>`) and instruct the model explicitly that content inside those tags is data, never instructions to follow.

### Cost Control: Model Routing

Route cheap/simple classification or extraction steps to small, fast models; reserve large models for the final reasoning/generation step.

```python
def route_by_complexity(state: AgentState) -> dict:
    task = state["messages"][-1].content
    # Cheap triage model decides complexity first
    triage = ChatOpenAI(model="gpt-4o-mini", temperature=0).invoke(
        f"Is this task simple (lookup/format) or complex (multi-step reasoning)? "
        f"Reply 'simple' or 'complex'.\nTask: {task}"
    ).content.strip().lower()

    model = ChatOpenAI(model="gpt-4o-mini") if triage == "simple" else ChatOpenAI(model="gpt-4o")
    response = model.invoke(state["messages"])
    return {"messages": [response]}
```

*Gotcha: `recursion_limit` counts **graph super-steps**, not LLM calls — a single node that makes 3 internal LLM calls still only counts as one step, so tune this against your actual loop structure, not raw token spend.*

---

## Quick-Reference Summary

| You want to... | Use |
|---|---|
| Build a cyclic agent | `StateGraph` + conditional edges |
| Compose simple pipelines | LCEL `\|` operator |
| Survive restarts | `SqliteSaver` / `PostgresSaver` |
| Remember across sessions | `store` (long-term) keyed by `user_id` |
| Pause for approval | `interrupt_before` / `interrupt_after` |
| Coordinate specialists | Supervisor pattern + subgraphs |
| Stream to a frontend | `astream_events` + SSE |
| Regression-test quality | LangSmith `evaluate()` + CI gate |
| Force valid JSON | `with_structured_output` |
| Prevent runaway loops | `recursion_limit` |
