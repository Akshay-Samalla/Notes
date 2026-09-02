# AI Agents: The Complete Study Guide

> A self-contained, practical guide for developers who already know how to call an LLM API and now want to build agents that actually work in production.

---

## Table of Contents

1. [What Is an AI Agent?](#part-1-what-is-an-ai-agent)
2. [The Core Components of an Agent](#part-2-the-core-components-of-an-agent)
3. [The 4 Core Agentic Design Patterns](#part-3-the-4-core-agentic-design-patterns)
4. [The AI Agents Stack (2026 Edition)](#part-4-the-ai-agents-stack-2026-edition)
5. [Agent Frameworks — A Practical Comparison](#part-5-agent-frameworks--a-practical-comparison)
6. [Memory in AI Agents](#part-6-memory-in-ai-agents)
7. [Tool Design & The Model Context Protocol (MCP)](#part-7-tool-design--the-model-context-protocol-mcp)
8. [Building Reliable Agents — Testing, Evals & Guardrails](#part-8-building-reliable-agents--testing-evals--guardrails)
9. [Multi-Agent Systems](#part-9-multi-agent-systems)
10. [Deployment & Production Considerations](#part-10-deployment--production-considerations)

---

## Part 1: What Is an AI Agent?

### The one-sentence definition

**An AI agent is an LLM wrapped in a loop.** That's it. Everything else — memory, planning, tools, multi-agent orchestration — is scaffolding built around one repeating cycle:

```
think → act → observe → repeat
```

The model looks at the current state of the world (the conversation, the tool outputs so far), decides what to do next, does it, looks at what happened, and decides again. It keeps doing this until it believes the task is finished, or until something (you) stops it.

Compare this to a normal LLM API call, which is a **single turn**: you send a prompt, you get a completion, you're done. An agent is what you get when you take that single turn and put it inside a `while` loop that keeps feeding the model's own actions and their results back in as new context.

### The evolution that got us here

It's worth walking through this lineage, because each step solved a specific limitation of the one before it:

- **Prompt → Response.** The original mode. You ask, it answers, from parametric memory only. No way to check its work, no way to touch the outside world.
- **Chain-of-Thought (CoT).** Researchers found that asking the model to "think step by step" before answering improved accuracy on reasoning tasks. This didn't add any new capability — it just gave the model room to reason on the page before committing to an answer. Still a single turn, still no tools.
- **ReAct (Reasoning + Acting).** The real turning point. ReAct interleaved reasoning traces with **actions** — the model could now emit a "Thought," then an "Action" (a tool call), receive an "Observation" (the tool's result), and continue reasoning from there. This is the direct ancestor of every modern agent loop.
- **Fully autonomous agents.** Take the ReAct loop, give it a longer leash (more iterations, more tools, persistent memory across sessions), and let it decide its own sub-goals rather than following a fixed script. This is where we are now.

**Engineer's Intuition:** The jump from CoT to ReAct is the single most important conceptual leap in this entire field. CoT makes the model *think* better. ReAct makes the model *act* on the world and *correct itself* based on what actually happened, rather than what it assumed would happen. Almost every agent framework you'll encounter — LangGraph, CrewAI, the raw OpenAI/Anthropic tool-calling loop — is a ReAct loop with different amounts of paint on top. If you deeply understand ReAct, you understand 80% of what "agent frameworks" are selling you.

### Agents vs. workflows — the distinction that actually matters

This is the single most useful mental model for deciding *whether you should even build an agent*:

| | **Workflow** | **Agent** |
|---|---|---|
| Control flow | Fixed, defined in code (`if`/`else`, a DAG) | Dynamic — the model decides the next step at runtime |
| Predictability | High — you can enumerate every path | Lower — paths emerge from model reasoning |
| Best for | Tasks with a known, repeatable shape | Open-ended tasks where the steps can't be fully enumerated in advance |
| Failure mode | Fails loudly at a known branch | Can loop, choose a bad tool, or drift off-task |
| Debuggability | Easy — trace the code path | Harder — trace the reasoning trace |

**Engineer's Intuition:** The biggest mistake teams make is reaching for "agent" when they actually need a workflow. If you can draw the flowchart of what should happen for 95% of your cases, *write that flowchart in code* and only call the LLM for the parts that genuinely require judgment. A hard-coded pipeline with one well-scoped LLM step is more reliable, cheaper, and easier to debug than a fully autonomous agent doing the same job. Reach for agents when the number of possible paths is too large to enumerate, or when the right sequence of actions genuinely depends on what earlier actions returned. Your L1 ticket classification pipeline, for example, is arguably a **workflow with an agentic classification step** — the routing logic (which folder a ticket lands in) is deterministic once a category and confidence score exist; the *agentic* part is only the reasoning that produces that category from an ambiguous ticket body.

### The agent loop — the atomic unit

Strip away every framework and here is what remains:

1. **Observe** the current state (system prompt + conversation history + last tool result).
2. **Reason** about what to do next (the model's internal "thought").
3. **Act** — either call a tool, or produce a final answer.
4. If a tool was called, **execute it**, capture the result, append it to the context, and go back to step 1.
5. Stop when the model produces a final answer (or you hit a safety limit).

### Code Example: A minimal agent loop, no frameworks

This is the raw loop underneath every agent framework you'll ever use. Read it line by line before you touch LangGraph or CrewAI — it will save you hours of framework-induced confusion later.

```python
"""
A minimal, framework-free agent loop using the OpenAI-style
chat completions API with function calling (tool use).
Swap `client.chat.completions.create` for the Anthropic Messages
API and it looks almost identical — the loop shape doesn't change.
"""

import json
from openai import OpenAI

client = OpenAI()

# --- Step 1: Define the tools the agent can use ---

def get_weather(city: str) -> str:
    # In real life this would hit a weather API.
    fake_data = {"Bengaluru": "28C, light rain", "Tokyo": "19C, clear"}
    return fake_data.get(city, "No data for that city")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

AVAILABLE_FUNCTIONS = {"get_weather": get_weather}

# --- Step 2: The loop itself ---

def run_agent(user_prompt: str, max_iterations: int = 8) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": user_prompt},
    ]

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
        )
        message = response.choices[0].message
        messages.append(message)

        # ACT: did the model choose to call a tool, or is it done?
        if not message.tool_calls:
            return message.content  # final answer -> exit the loop

        # OBSERVE: execute every requested tool call, feed results back in
        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            result = AVAILABLE_FUNCTIONS[fn_name](**fn_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })
        # loop back to step 1 with the new observation in context

    return "Stopped: max iterations reached without a final answer."


if __name__ == "__main__":
    print(run_agent("What's the weather like in Bengaluru right now?"))
```

Notice what's *not* here: no framework, no special "agent" class, no magic. Just a `for` loop, a list of messages that grows, and a branch that checks "did the model ask for a tool, or is it done?" Every agent framework you'll meet in Part 5 is this loop with extra structure (typed state, graphs, retries, observability) wrapped around it.


---

## Part 2: The Core Components of an Agent

Every agent, however fancy the framework, is assembled from the same six pieces. Frameworks differ mainly in how much of each piece they build for you versus leave to you.

### 1. Reasoning Engine (The LLM)

This is the "brain" — the model that reads the current state and decides what happens next. Model choice is not a minor detail; it drives cost, latency, and reliability more than almost any other decision you'll make.

**Engineer's Intuition:** Don't reflexively reach for the biggest, smartest model for every step of an agent. A common production pattern is to use a strong reasoning model (e.g., a frontier model) for planning and tool-argument generation — the steps where mistakes compound — and a cheap, fast model for classification, summarization, or formatting steps where the task is narrow and well-specified. Your own L1 classification agent is a good example of this trade-off in practice: running inference locally via Microsoft Foundry Local trades some raw capability for latency, cost, and data-locality guarantees that matter a lot for ticket data that shouldn't leave your infrastructure.

### 2. Tool Use / Function Calling

Tools are how the agent touches the world outside its own context window — APIs, databases, file systems, code execution, search. Function calling is the mechanism: you describe a function's name, purpose, and input schema in JSON Schema, the model emits a structured call to that schema when it wants to use the tool, and your code executes it and returns the result as text back into the context.

The model **never actually executes anything**. It only ever emits *a request to call a function with certain arguments*. Execution, sandboxing, and error handling are entirely your responsibility as the engineer. This single fact is the root of most agent security and reliability work in Part 8.

### 3. Planning

Planning is the process of decomposing a high-level, ambiguous goal ("triage this backlog of 200 tickets") into a sequence of smaller, executable sub-tasks ("classify ticket 1", "check if it matches a known incident pattern", "assign confidence", "route or flag for review"). Planning can be:

- **Implicit** — the model just reasons step-by-step within the ReAct loop, deciding the next action based on what's happened so far (no explicit plan is ever written down).
- **Explicit** — the model is first asked to output a plan (a numbered list of steps) *before* execution begins, and that plan is then followed (and potentially revised) as sub-tasks complete.

**Engineer's Intuition:** Explicit planning trades some flexibility for a lot of debuggability. If your agent writes down "Step 1: search for X, Step 2: verify against Y, Step 3: summarize" before it does anything, you get a natural checkpoint to show a human, log, or gate on approval — which matters enormously in an infra/ops context where "the agent silently decided to do something" is a much scarier failure mode than "the agent proposed a plan I could review."

### 4. Reflection

Reflection is the agent looking at its own intermediate output — a draft, a tool result, a partial answer — and critiquing it before proceeding. It's the mechanism that turns "generate once and hope" into "generate, check, revise." Reflection can be done by the same model in a second pass, or by a *separate* "critic" call/agent, which tends to produce more honest criticism because the critic isn't anchored on defending its own first draft.

### 5. Memory

Memory is what lets an agent's reasoning persist beyond a single context window — across steps within a task (short-term) and across sessions entirely (long-term). Covered in full depth in Part 6, but the component-level point is: without memory, every loop iteration only knows what's in the current message list, and every new session starts from zero. Memory is what turns a one-shot tool-user into something that improves, remembers preferences, or resumes interrupted work — like a RAG store of previously classified tickets that a new ticket can be compared against for similarity.

### 6. Orchestration

Orchestration is the glue: the code that sequences the other five components into an actual running loop — deciding when to call the reasoning engine, when to invoke a tool, when to write to memory, when to reflect, and when to stop. In a raw implementation (like Part 1's code example), orchestration is just your `for` loop. In a framework like LangGraph, orchestration is an explicit graph of nodes and edges you define.

### Code Example: Building up component by component

```python
"""
Progressive build: start with a 'dumb' loop (no tools),
add tool use, then add a simple memory layer.
Uses the Anthropic Messages API shape for variety.
"""

import anthropic

client = anthropic.Anthropic()

# --- Stage 1: dumb loop, no tools, no memory ---
def dumb_agent(prompt: str) -> str:
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


# --- Stage 2: add a tool (a calculator) ---
TOOLS = [{
    "name": "calculator",
    "description": "Evaluate a basic arithmetic expression",
    "input_schema": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
}]

def calculator(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"error: {e}"

def agent_with_tools(prompt: str, max_iterations: int = 5) -> str:
    messages = [{"role": "user", "content": prompt}]
    for _ in range(max_iterations):
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        tool_results = []
        for block in resp.content:
            if block.type == "tool_use" and block.name == "calculator":
                result = calculator(**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        messages.append({"role": "user", "content": tool_results})

    return "Stopped: max iterations reached."


# --- Stage 3: add a simple persistent memory layer ---
import sqlite3

def remember(user_id: str, fact: str):
    conn = sqlite3.connect("agent_memory.db")
    conn.execute("CREATE TABLE IF NOT EXISTS memory (user_id TEXT, fact TEXT)")
    conn.execute("INSERT INTO memory VALUES (?, ?)", (user_id, fact))
    conn.commit()
    conn.close()

def recall(user_id: str) -> list[str]:
    conn = sqlite3.connect("agent_memory.db")
    conn.execute("CREATE TABLE IF NOT EXISTS memory (user_id TEXT, fact TEXT)")
    rows = conn.execute("SELECT fact FROM memory WHERE user_id=?", (user_id,)).fetchall()
    conn.close()
    return [r[0] for r in rows]

def agent_with_memory(user_id: str, prompt: str) -> str:
    prior_facts = recall(user_id)
    system = "Known facts about this user:\n" + "\n".join(prior_facts) if prior_facts else ""
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text
```

Each stage adds exactly one component. This is the recommended way to build your *first* real agent too: get the dumb loop working, add one tool and prove the round-trip, then add memory — never all three at once.


---

## Part 3: The 4 Core Agentic Design Patterns

These patterns (popularized in Andrew Ng's writing on agentic workflows) are not mutually exclusive frameworks — they're composable techniques. A production agent typically uses more than one at once.

### Pattern 1 — Reflection

The agent produces an output, then critiques its own work, then revises. This is the single highest-leverage pattern for improving output *quality* without changing the underlying model, because it exploits an asymmetry: models are often better at *evaluating* a candidate answer than *generating* a perfect one on the first try.

**Engineer's Intuition:** Reflection roughly doubles your token cost and latency for a given task (generate + critique + revise ≈ 2-3x the calls of a single pass). Use it selectively — on outputs where correctness matters and errors are expensive (a customer-facing summary, a code change) — not on every single step of every agent.

### Pattern 2 — Tool Use

Already covered in depth in Part 2. As a *pattern*, the emphasis is: tools are how an agent's capabilities scale beyond "whatever the model memorized during training." A model that can call a search API, a database, or a code interpreter is categorically more capable than the same model with no tools, regardless of how large or smart it is.

### Pattern 3 — Planning

Also introduced in Part 2. As a pattern, planning is what separates "an agent that reacts one step at a time" from "an agent that has a coherent strategy for a multi-step goal." The key production nuance: plans should be **revisable**, not fixed. A good planning pattern re-evaluates the plan after each major step, rather than blindly executing a plan made before any information was gathered.

### Pattern 4 — Multi-Agent Collaboration

Instead of one agent trying to do everything, split the work across multiple specialized agents that each have a narrow role, then have them hand off work to each other — the way a human team splits a project across a researcher, a writer, and an editor. Detailed in full in Part 9.

**Engineer's Intuition:** Multi-agent systems are often reached for too early. The honest reason to go multi-agent is *context isolation* (each agent's context window only contains what's relevant to its narrow job, which improves focus and reduces token cost) or *role specialization* (different system prompts / tools / models per role), not because "multi-agent" sounds more sophisticated. A single well-designed agent with good tools often outperforms a poorly-decomposed multi-agent system.

### Code Example: Reflection in isolation

```python
"""
An agent that writes a short piece of code, critiques its own
output against a checklist, and revises once if the critique
finds a real problem.
"""

import anthropic

client = anthropic.Anthropic()

def generate(task: str) -> str:
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": f"Write a Python function for: {task}"}],
    )
    return resp.content[0].text


def critique(task: str, draft: str) -> str:
    prompt = (
        f"Task: {task}\n\nDraft solution:\n{draft}\n\n"
        "Critique this draft against: correctness, edge cases, "
        "and readability. If it's genuinely fine, reply exactly "
        "'NO ISSUES'. Otherwise list concrete problems."
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def revise(task: str, draft: str, feedback: str) -> str:
    prompt = (
        f"Task: {task}\n\nDraft:\n{draft}\n\nFeedback:\n{feedback}\n\n"
        "Produce a revised version that addresses the feedback."
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def reflective_agent(task: str) -> str:
    draft = generate(task)
    feedback = critique(task, draft)
    if "NO ISSUES" in feedback:
        return draft
    return revise(task, draft, feedback)


if __name__ == "__main__":
    print(reflective_agent("parse a CSV of ticket IDs and return duplicate IDs"))
```


---

## Part 4: The AI Agents Stack (2026 Edition)

Think of this as the vertical slice of technology you touch when you go from "a model" to "a deployed agent." Each layer solves a distinct problem, and vendors compete at each layer independently — you don't have to buy all your layers from the same company.

| Layer | Problem it solves | How to evaluate options |
|---|---|---|
| **1. Models & Inference** | Raw reasoning and generation capability | Accuracy on tasks like yours, latency, cost per token, context window size, whether it can run locally/on-prem if data locality matters |
| **2. Protocols & Tools** | How an agent discovers and calls external capabilities in a standardized way | Does the ecosystem you need (databases, SaaS tools, internal systems) already expose an MCP server? How much custom glue code will you write regardless? |
| **3. Memory & Knowledge** | Persisting state and retrievable knowledge across steps and sessions | Latency of retrieval, whether you need exact recall vs. semantic similarity, cost of the vector store at your data scale |
| **4. Frameworks** | Orchestration boilerplate — the loop, state management, retries | How much control do you need vs. how much boilerplate do you want handled for you? (Part 5) |
| **5. Observability & Evaluation** | Knowing *why* an agent did what it did, and whether it's getting better or worse over time | Can you see the full reasoning trace? Can you replay a failed run? Can you compute a success-rate metric automatically? |
| **6. Guardrails & Safety** | Constraining what the agent is *allowed* to do, independent of what the model *decides* to do | Are constraints enforced in code (hard) or only via prompting (soft)? Can a guardrail block an action before it executes? |

**Engineer's Intuition:** Layers 1-4 get almost all the attention in tutorials and demos because they're what makes an agent *work at all*. Layers 5 and 6 are what make an agent *safe to run unattended in production*, and they're consistently underbuilt. If you're building something like your L1 ticket router — something that acts on real tickets with real downstream consequences — treat observability and guardrails as first-class requirements from day one, not something you bolt on after a demo succeeds. A demo that works once tells you almost nothing about layer 5 and 6 concerns; those only reveal themselves at volume, over time, and under adversarial or malformed input.

A useful shorthand: **Layers 1–4 make the agent capable. Layers 5–6 make the agent trustworthy.** You need both to ship something real.


---

## Part 5: Agent Frameworks — A Practical Comparison

Frameworks exist to remove boilerplate from the raw loop you saw in Part 1: state management, retries, streaming, tool-schema generation, and (in some cases) a visual/no-code layer. None of them do anything the raw loop couldn't do — they trade some control for less code you have to write and maintain yourself.

| Framework | Model | License / ownership | Best for |
|---|---|---|---|
| **LangChain / LangGraph** | Code-first; you define a graph of nodes and edges representing agent state transitions | MIT-licensed, open source — you own and can self-host everything | Teams that want fine-grained control over the loop's control flow and are comfortable writing Python |
| **AutoGPT** | Agent *platform* — build and configure agents largely without writing orchestration code yourself | Hosted runtime with a dashboard | Fast prototyping, non-engineers configuring simple autonomous agents |
| **Microsoft Agent Framework** | Successor to Semantic Kernel and AutoGen, unified under one SDK; models agent orchestration as a data-flow graph | Azure-native, deep integration with the Microsoft ecosystem | Teams already standardized on Azure/.NET or Semantic Kernel who want first-party support and Azure-native deployment |
| **CrewAI** | Role-based multi-agent orchestration — you define "crews" of agents with roles, goals, and backstories that collaborate on a shared task | Open source | Multi-agent workflows where the "team of specialists" mental model maps cleanly onto your problem |

### Decision framework: which one should you actually use?

Ask three questions, in this order:

1. **How much customization do I need over the control flow?** If you need to intervene mid-loop, add custom retry logic, or branch on conditions that don't map to a framework's built-in primitives, favor code-first tools (LangGraph) over platforms (AutoGPT) — platforms optimize for ease of configuration at the cost of exactly this kind of control.
2. **What's my team's expertise?** A team of strong Python engineers gets more value from LangGraph's explicit graph model than from a low-code platform's UI. A team without dedicated ML/AI engineers may ship faster on a hosted platform, accepting less control in exchange.
3. **How much vendor lock-in can I tolerate?** Azure-native tooling (Microsoft Agent Framework) is an excellent choice *if* you're already deep in that ecosystem — but it raises your switching cost later. Open, self-hosted, MIT-licensed tools (LangGraph, CrewAI) minimize lock-in at the cost of doing more integration work yourself.

**Engineer's Intuition:** For infra/ops-flavored agents — the kind that touch tickets, incidents, change management systems — favor frameworks you can self-host and fully audit over hosted platforms, even if the hosted platform is faster to prototype in. You need to be able to answer "exactly what did this agent do and why" for any given action, and that's much easier when you control every layer of the stack, as you already do with your ChromaDB + LangChain ReAct + local-inference setup.

### Code Example: The same simple agent in two frameworks

**Same task in both:** a research assistant that takes a query, "searches" (mocked here), and summarizes the result.

**LangGraph version:**

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-6")

class AgentState(TypedDict):
    query: str
    search_result: str
    summary: str

def search_node(state: AgentState) -> AgentState:
    # In real life: call a search API here.
    state["search_result"] = f"[mock search results for: {state['query']}]"
    return state

def summarize_node(state: AgentState) -> AgentState:
    resp = llm.invoke(f"Summarize this in 2 sentences:\n{state['search_result']}")
    state["summary"] = resp.content
    return state

graph = StateGraph(AgentState)
graph.add_node("search", search_node)
graph.add_node("summarize", summarize_node)
graph.set_entry_point("search")
graph.add_edge("search", "summarize")
graph.add_edge("summarize", END)

app = graph.compile()
result = app.invoke({"query": "latest Azure Firewall pricing changes"})
print(result["summary"])
```

**CrewAI version:**

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Researcher",
    goal="Find accurate, current information on the given topic",
    backstory="An expert at locating and validating information quickly.",
)

writer = Agent(
    role="Writer",
    goal="Summarize research findings into a concise 2-sentence brief",
    backstory="A technical writer who values brevity and precision.",
)

research_task = Task(
    description="Research: latest Azure Firewall pricing changes",
    expected_output="Raw findings, unsummarized",
    agent=researcher,
)

summarize_task = Task(
    description="Summarize the research findings in 2 sentences",
    expected_output="A 2-sentence summary",
    agent=writer,
    context=[research_task],
)

crew = Crew(agents=[researcher, writer], tasks=[research_task, summarize_task])
result = crew.kickoff()
print(result)
```

**The difference in one sentence:** LangGraph makes you think in terms of *state and transitions between nodes* (a data-flow mental model); CrewAI makes you think in terms of *roles and tasks handed between team members* (an org-chart mental model). Same underlying ReAct-style loop under the hood in both cases — different abstractions on top.


---

## Part 6: Memory in AI Agents

### Short-term memory

This is simply "what's currently in the context window" — the running conversation, the tool calls and results from the current task. It disappears the moment the session ends (or the context window fills up and gets truncated/summarized). Every agent has this by default; it costs you nothing extra to implement because it's just... the message list.

### Long-term memory

Anything that needs to survive *past* the current session — user preferences, prior decisions, a knowledge base the agent can query. Long-term memory is almost always implemented as an external store the agent queries via a tool call, not as something magically "inside" the model:

- **Vector databases** (Chroma, Pinecone, Weaviate, pgvector) — store embeddings of text chunks, retrieve by semantic similarity. Best for "find things related in meaning to X" — exactly the pattern your L1 agent's ChromaDB-based RAG similarity search uses to find prior tickets like the current one.
- **Relational stores (SQLite, Postgres)** — best for exact, structured recall: "what did this user say their preference was," "what's the status of ticket #4821." Cheap, simple, and often underrated compared to vector stores for facts that don't need semantic fuzziness.
- **External document/knowledge stores** — full documents or knowledge bases the agent can search or fetch from, often combined with a vector index for retrieval.

**Engineer's Intuition:** A very common mistake is reaching for a vector database when a plain SQL table would do. Vector search solves "find semantically similar things" — it is the wrong tool for "look up an exact fact by ID." Your ticket classification system correctly uses both: exact routing logic and thresholds live in code/config, while ChromaDB is reserved for the genuinely fuzzy problem (does this new ticket resemble one we've seen before).

### State management

For agents that run long, multi-step tasks — especially ones that might be interrupted, retried, or resumed — you need **checkpointing**: periodically persisting the agent's full state (messages so far, intermediate variables) somewhere durable, so a crash or restart doesn't lose the work. LangGraph, for example, supports checkpointers that can write state to Redis, SQLite, or Postgres after every step, letting you resume a graph exactly where it left off.

| Memory type | Lifespan | Typical backing store | Use for |
|---|---|---|---|
| Short-term | Single session/task | In-context (message list) | The immediate task at hand |
| Long-term (semantic) | Across sessions | Vector DB (Chroma, Pinecone) | "Find things like X" |
| Long-term (exact) | Across sessions | SQL / key-value store | User prefs, structured facts, audit trail |
| State checkpoint | Across restarts of the *same* task | Redis / Postgres via a checkpointer | Resuming a long-running or interrupted agent run |

### Code Example: Remembering user preferences across sessions with SQLite

```python
"""
A minimal agent that persists user preferences in SQLite and
uses them to personalize behavior in future sessions —
independent of any framework.
"""

import sqlite3
import anthropic

client = anthropic.Anthropic()
DB_PATH = "preferences.db"

def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS preferences (user_id TEXT, key TEXT, value TEXT, "
        "PRIMARY KEY (user_id, key))"
    )
    return conn

def set_preference(user_id: str, key: str, value: str) -> None:
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO preferences VALUES (?, ?, ?)", (user_id, key, value)
    )
    conn.commit()
    conn.close()

def get_preferences(user_id: str) -> dict:
    conn = _conn()
    rows = conn.execute(
        "SELECT key, value FROM preferences WHERE user_id=?", (user_id,)
    ).fetchall()
    conn.close()
    return dict(rows)

# The agent decides, via a tool call, when something is worth remembering.
REMEMBER_TOOL = {
    "name": "remember_preference",
    "description": "Store a durable user preference for future sessions",
    "input_schema": {
        "type": "object",
        "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
        "required": ["key", "value"],
    },
}

def run_session(user_id: str, prompt: str) -> str:
    known = get_preferences(user_id)
    system = (
        "Known preferences for this user: " + str(known) if known else
        "No known preferences for this user yet."
    )

    messages = [{"role": "user", "content": prompt}]
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=500, system=system,
        tools=[REMEMBER_TOOL], messages=messages,
    )

    for block in resp.content:
        if block.type == "tool_use" and block.name == "remember_preference":
            set_preference(user_id, block.input["key"], block.input["value"])

    return "".join(b.text for b in resp.content if b.type == "text")
```

Run this twice with the same `user_id` and a preference mentioned in the first call, and you'll see the second call's `system` prompt already contains it — memory surviving the process restart entirely through SQLite, no framework required.


---

## Part 7: Tool Design & The Model Context Protocol (MCP)

### What makes a good tool

Tool design is prompt engineering for a machine reader (the model) rather than a human one, and it matters more than most people expect — a poorly-described tool causes wrong arguments, wrong tool selection, and silent failures far more often than the underlying model being "not smart enough."

- **Clear description.** Say what the tool does, when to use it, and — just as important — when *not* to use it. "Searches internal ticket history for similar past incidents" is far better than "search."
- **Well-defined input schema.** Use JSON Schema types, required fields, enums for constrained values, and descriptions on every parameter. If a parameter is a date, say what format. If it's an ID, say what kind of ID.
- **Idempotency where possible.** A tool that's safe to call twice with the same arguments (e.g., "get ticket status") is much easier to retry safely than one with side effects (e.g., "close ticket"). Where a tool *does* have side effects, make that unambiguous in its description, and consider requiring a confirmation step before it executes (see Part 8's guardrails).

**Engineer's Intuition:** Treat every tool description as documentation written for a very literal-minded new hire who has never seen your system before and will follow the words exactly. Vague tool descriptions are the single most common root cause of "the agent called the wrong tool" bugs — far more common than the model failing to reason correctly given good information.

### MCP — the Model Context Protocol

MCP is the open standard that emerged to solve tool *interoperability*: instead of every framework and every application writing its own custom integration for every tool/data source, an MCP server exposes tools, resources, and prompts in a standard way that any MCP-compatible client (any agent framework, any application) can consume without custom glue code. Think of it as roughly analogous to what a REST/OpenAPI spec did for web APIs, but purpose-built for LLM tool use — and increasingly the norm by which vendors (databases, SaaS products, internal platforms) expose themselves to agents at all, rather than each agent framework needing a bespoke integration per tool.

**Engineer's Intuition:** The practical payoff of MCP isn't really felt when you're calling one or two tools you wrote yourself — it shows up once you're integrating with a dozen external systems (ticketing platforms, cloud provider APIs, internal databases) and would otherwise be writing and maintaining a dozen bespoke integrations. If an ecosystem you already depend on ships an MCP server, prefer it over a hand-rolled integration; you get maintenance, auth handling, and schema correctness for free from whoever maintains that server.

### Error handling — what happens when a tool fails

Tools *will* fail — a network timeout, a malformed argument, a downstream API rate limit. How you handle this determines whether your agent degrades gracefully or fails opaquely:

- **Retry logic.** Transient failures (timeouts, 5xx errors) should be retried with backoff, invisibly to the model, *before* the failure is ever surfaced as an observation.
- **Fallbacks.** If a tool consistently fails, consider a fallback tool or a degraded response ("I couldn't reach the live pricing API — here's the cached value from this morning") rather than letting the agent loop forever or hallucinate an answer.
- **Graceful degradation, surfaced as an observation.** When a tool genuinely fails and there's no fallback, return a clear error string as the tool result (not an exception that crashes your loop) so the model can reason about it — try a different tool, ask the user for more info, or report the limitation honestly.

### Code Example: Designing and registering a small toolset

```python
"""
Three tools with production-grade descriptions: web search,
database query, and a calculator — plus basic error handling.
"""

TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the public web for current information. Use this for "
            "facts that may have changed recently or that you don't know. "
            "Do NOT use for internal company data — use database_query instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query, 2-6 words"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "database_query",
        "description": (
            "Run a read-only SQL SELECT query against the internal tickets "
            "database. Only SELECT statements are allowed — write operations "
            "will be rejected. Use this for exact lookups by ticket ID, "
            "status, or assignee."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "A single SELECT statement"}
            },
            "required": ["sql"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression. Use for any numeric computation rather than doing math yourself.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]


def database_query(sql: str) -> str:
    if not sql.strip().lower().startswith("select"):
        return "ERROR: only SELECT statements are permitted."
    try:
        # Execute against a real read-only connection in production.
        return f"[mock result set for: {sql}]"
    except Exception as e:
        return f"ERROR: query failed: {e}"


def calculator(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"ERROR: could not evaluate expression: {e}"


def web_search(query: str) -> str:
    try:
        # Call a real search API in production; this is a stub.
        return f"[mock search results for: {query}]"
    except Exception as e:
        return f"ERROR: search failed, try again or use a different tool: {e}"


DISPATCH = {
    "web_search": web_search,
    "database_query": database_query,
    "calculator": calculator,
}
```

Notice the `database_query` description explicitly forbids write operations *and* the implementation enforces it in code — never rely on the prompt alone to prevent a destructive action a tool is technically capable of performing. That enforcement belongs in Part 8's guardrails, and this is exactly where it starts: at the tool boundary.


---

## Part 8: Building Reliable Agents — Testing, Evals & Guardrails

### Why agents fail

In practice, almost every agent failure traces back to one of a small number of root causes:

- **The loop never stops.** The model keeps calling tools, revising, or "double-checking" indefinitely because nothing forces it to converge on a final answer, and you have no hard iteration cap.
- **Wrong tool arguments.** The model calls the right tool with malformed, hallucinated, or subtly wrong arguments — often because the tool's input schema or description was ambiguous (see Part 7).
- **No plan for what happens after a result comes back.** The agent gets a tool result but has no strategy for handling an *unexpected* result (an empty search, an error string, a value outside the expected range) and either loops, hallucinates a plausible-sounding continuation, or crashes your orchestration code.

**Engineer's Intuition:** Notice that none of these three root causes are "the model isn't smart enough." They are almost all *engineering* failures — missing limits, missing validation, missing branches for the unhappy path — the same categories of bugs you'd guard against in any distributed system. Treat agent reliability as a systems engineering problem first and a prompting problem second.

### Testing

- **Unit tests for tools.** Each tool is a normal function — test it exactly like you'd test any function: valid inputs, edge cases, malformed inputs, and what it returns when the thing it calls fails.
- **Integration tests for the full loop.** Feed the agent a fixed prompt and assert on the *shape* of the outcome (did it call the expected tool, did it terminate within N iterations, does the final answer contain expected key facts) rather than asserting on exact text, since LLM outputs are non-deterministic by nature.

### Evaluations

Evals are how you measure whether an agent is *actually* good at its job, not just "worked in the one demo I watched." At minimum, track:

- **Success rate** — did the agent achieve the intended outcome, measured against a labeled test set of realistic tasks (for your ticket classifier: does the predicted category match a human-labeled ground truth?).
- **Cost** — tokens and dollars per task, since agentic loops can silently burn far more tokens than a single completion.
- **Latency** — wall-clock time per task, especially important for anything a human is waiting on synchronously.

**Engineer's Intuition:** Build your eval set from *real* failure cases as you find them, not just from cases you imagined up front. Every bug you fix in production should also become a permanent regression test in your eval set — this is exactly how you avoid re-breaking the same edge case (a weirdly formatted ticket, an ambiguous incident description) six months later after an unrelated prompt change.

### Guardrails

Guardrails are **hard constraints enforced in code**, independent of what the model decides — the whole point is that they hold even if the model reasons its way into a bad decision. Contrast this with "instructing the model not to do X in the prompt," which is a *soft* constraint the model can still violate.

- **Max iterations** — a hard cap on loop iterations, so a runaway agent stops instead of burning tokens indefinitely.
- **Tool call validators** — code that checks a tool call's arguments *before* execution (e.g., reject a SQL query that isn't a SELECT, reject a file path outside an allowed directory) and refuses to execute if the check fails, returning an error observation to the model instead.
- **Human-in-the-loop approval gates** — for genuinely destructive or high-stakes actions (closing a ticket, modifying infrastructure), require explicit human confirmation before the tool actually executes, regardless of how confident the model is.

### Code Example: A max-iterations guardrail and a tool-call validator

```python
"""
Two guardrails layered onto the raw loop from Part 1:
1. A hard iteration cap.
2. A validator that runs before any tool executes.
"""

MAX_ITERATIONS = 6

def validate_tool_call(name: str, args: dict) -> str | None:
    """Return an error string if the call should be blocked, else None."""
    if name == "database_query":
        sql = args.get("sql", "").strip().lower()
        if not sql.startswith("select"):
            return "BLOCKED: only SELECT statements are permitted."
        if "drop " in sql or "delete " in sql:
            return "BLOCKED: destructive keywords are never permitted, even inside a SELECT."
    if name == "close_ticket":
        return "BLOCKED: closing a ticket requires human approval. Ask the user to confirm first."
    return None


def guarded_agent_loop(messages: list, tools: list, dispatch: dict) -> str:
    for iteration in range(MAX_ITERATIONS):
        response = call_model(messages, tools)  # your model call here
        messages.append(response.message)

        if not response.tool_calls:
            return response.text  # done

        for call in response.tool_calls:
            block_reason = validate_tool_call(call.name, call.args)
            if block_reason:
                result = block_reason  # feed the block reason back as the observation
            else:
                result = dispatch[call.name](**call.args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})

    return "Stopped: max iterations reached without a final answer. Escalating to a human."
```

The critical detail: `validate_tool_call` runs **before** `dispatch[call.name](**call.args)` — the model is *told* about a blocked action via the observation, not silently overruled, but it can never actually cause the blocked side effect. This is the difference between a guardrail and a suggestion.


---

## Part 9: Multi-Agent Systems

### Why multiple agents

A single agent with a huge system prompt, a dozen tools, and a broad mandate tends to get *worse*, not better, as you keep adding responsibility to it — the context window fills with irrelevant instructions for whatever the current sub-task is, and the model has to hold far more in mind at once. Splitting a large task across multiple narrowly-scoped agents mirrors why human teams specialize: a researcher and a writer each do their one job well, rather than one person context-switching between fundamentally different modes of thinking.

**Engineer's Intuition:** The real benefit is **context isolation**, not "more agents = smarter." An agent with a tight, single-purpose system prompt and three relevant tools reliably outperforms a generalist agent with thirty tools and a sprawling prompt trying to do the same overall job, because the generalist's context is diluted with instructions that don't apply to the step it's currently on.

### Orchestration patterns

- **Hub-and-spoke (central orchestrator).** One orchestrator agent (or plain code) receives the task, delegates sub-tasks to specialist agents, and assembles their results. The specialists don't talk to each other directly — everything flows through the hub. Easiest to reason about and debug.
- **Supervisor (manager delegates).** Similar to hub-and-spoke, but the "manager" is itself an LLM making dynamic delegation decisions at runtime (which specialist to call next, in what order) rather than following fixed code — a supervisor pattern is essentially "ReAct where the tools are other agents."
- **Peer-to-peer (agents negotiate).** Agents communicate directly with each other, potentially without a central coordinator, to jointly work out a solution. Most flexible, hardest to debug and constrain — reserve for cases where the interaction pattern genuinely can't be centralized.

**Engineer's Intuition:** Start with hub-and-spoke every time, even if you think you'll eventually need peer-to-peer. It's the easiest pattern to add logging, guardrails, and human checkpoints to, because every piece of communication passes through one place you control. Only graduate to a supervisor or peer-to-peer pattern once you've hit a concrete limitation hub-and-spoke can't express — don't reach for the more complex pattern speculatively.

### Handoffs

A handoff is the mechanism by which one agent passes a task (and the relevant context) to another — typically implemented as the delegating agent producing structured output (which agent to hand off to, and what context/instructions to pass) that your orchestration code then routes accordingly. The design question that matters most: **how much context follows the handoff?** Pass too little and the receiving agent lacks what it needs; pass too much and you've defeated the point of context isolation in the first place. The right amount is usually "the task description plus only the specific facts the next agent needs," not the entire conversation history.

### Code Example: A two-agent Researcher → Writer system

```python
"""
Two specialized agents with a hub-and-spoke handoff:
Researcher gathers information, Writer synthesizes a report.
The orchestrator (plain Python) controls the handoff explicitly.
"""

import anthropic

client = anthropic.Anthropic()

RESEARCHER_SYSTEM = (
    "You are a Researcher. Your only job is to gather accurate, relevant "
    "facts on the given topic using your search tool. Do not write prose "
    "summaries or reports -- just return structured findings as bullet points."
)

WRITER_SYSTEM = (
    "You are a Writer. You receive raw research findings and turn them "
    "into a clear, well-organized report for a technical audience. "
    "You do not do research yourself -- work only from what's given to you."
)

def mock_search(query: str) -> str:
    return f"- Finding 1 about {query}\n- Finding 2 about {query}\n- Finding 3 about {query}"

RESEARCH_TOOLS = [{
    "name": "search",
    "description": "Search for information on a topic",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}]

def researcher_agent(topic: str) -> str:
    messages = [{"role": "user", "content": f"Research this topic: {topic}"}]
    for _ in range(4):
        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=600,
            system=RESEARCHER_SYSTEM, tools=RESEARCH_TOOLS, messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")
        tool_results = []
        for block in resp.content:
            if block.type == "tool_use" and block.name == "search":
                result = mock_search(**block.input)
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
        messages.append({"role": "user", "content": tool_results})
    return "Researcher stopped: max iterations reached."


def writer_agent(findings: str) -> str:
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=800, system=WRITER_SYSTEM,
        messages=[{"role": "user", "content": f"Write a report from these findings:\n{findings}"}],
    )
    return resp.content[0].text


def research_and_write(topic: str) -> str:
    # This IS the handoff: the orchestrator explicitly passes only the
    # researcher's findings -- not the researcher's full message history --
    # into the writer's context.
    findings = researcher_agent(topic)
    report = writer_agent(findings)
    return report


if __name__ == "__main__":
    print(research_and_write("cost trends for Azure Application Gateway in 2026"))
```

Notice the orchestrator (`research_and_write`) is plain Python, not an LLM call — this is a hub-and-spoke pattern with a *code* hub, the simplest and most debuggable version of multi-agent collaboration. A supervisor pattern would replace that function body with an LLM call that decides dynamically whether to invoke the researcher again, skip straight to the writer, or bring in a third specialist — strictly more powerful, and strictly harder to reason about.


---

## Part 10: Deployment & Production Considerations

### Stateless vs. stateful

- **Stateless tool callers** treat every request independently — no memory of prior requests, no session concept. Simple to scale horizontally (any server can handle any request) and simple to reason about, but they can't personalize or resume interrupted work.
- **Multi-session agents that learn over time** carry state (via the long-term memory patterns from Part 6) across requests, tied to a user or task ID. More capable, but now you must think about where that state lives, how it's kept consistent under concurrent requests, and what happens if a request fails partway through a multi-step update to that state.

**Engineer's Intuition:** Default to stateless wherever the task allows it — it's a much smaller surface area of things that can go wrong, and it scales trivially. Only add statefulness where the *task itself* genuinely requires memory across requests (a conversation that spans multiple messages, a workflow that resumes after interruption). Don't add session state "because it seems more sophisticated" if a given request can be fully answered from its own inputs.

### Observability

You need to be able to answer, for any past agent run: what did it do, in what order, with what tool arguments, and why (what was the model's reasoning at each step)? This is **tracing** — capturing every step of the loop, not just the final output. LangSmith (paired naturally with LangChain/LangGraph) and OpenTelemetry-based tracing (framework-agnostic, integrates with whatever observability stack you already run — Datadog, Grafana, etc.) are the two common approaches. If you're already running an OpenTelemetry-based observability stack for your infrastructure, extending it to instrument agent traces keeps everything in one place rather than adding a second, agent-specific observability tool to operate.

**Engineer's Intuition:** Build tracing in from the start, not after your first production incident caused by an agent. The single most common regret teams report after their first agent-related production surprise is "we had no idea what it actually did until we went and reconstructed it from logs by hand." A trace of every tool call and its arguments turns a multi-hour incident investigation into a five-minute log lookup.

### Scaling

Handling concurrent agent requests is mostly a familiar backend-engineering problem — the agentic-specific wrinkle is that a single agent *task* can involve many sequential LLM calls (not one request/response), so your effective concurrency and cost model needs to account for the *number of model calls per task*, not just the number of incoming tasks. Rate limits from your model provider, and the cost of running many agent loops in parallel, become real capacity-planning inputs.

### Cost optimization

- **Caching.** Cache tool results that don't change often (a static lookup, a document that rarely updates) instead of re-fetching every loop iteration. Some model providers also support prompt caching for large, repeated system prompts/context, which materially cuts cost for agents with long, mostly-static instructions.
- **Right-sizing the model per step.** As noted in Part 2, use a cheaper/faster model for narrow, well-specified steps (classification, extraction, formatting) and reserve your most capable model for the steps that genuinely require deep reasoning (planning, ambiguous judgment calls).

### Code Example: Deploying an agent as a FastAPI endpoint with session state

```python
"""
A minimal FastAPI service exposing a stateful chat agent.
Session state (message history) is kept per session_id in memory
here for simplicity -- swap the dict for Redis/Postgres in production.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import anthropic
import uuid

app = FastAPI()
client = anthropic.Anthropic()

# In production: replace with a real store (Redis, Postgres) keyed by session_id.
SESSIONS: dict[str, list] = {}

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or str(uuid.uuid4())
    history = SESSIONS.get(session_id, [])
    history.append({"role": "user", "content": req.message})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=history,
    )
    reply_text = response.content[0].text
    history.append({"role": "assistant", "content": reply_text})

    SESSIONS[session_id] = history  # persist the growing session state
    return ChatResponse(session_id=session_id, reply=reply_text)

@app.get("/health")
def health():
    return {"status": "ok"}

# Run with: uvicorn this_file:app --host 0.0.0.0 --port 8000
```

Every earlier concept in this guide shows up here in miniature: the `/chat` endpoint is the agent loop's entry point, `SESSIONS` is a (deliberately naive) long-term memory layer, and the `/health` endpoint is the beginning of the observability you'd expand with real tracing before this went anywhere near production traffic.

---

## Closing: what to build first

If you take one thing from this guide, take this build order — it's the same progressive-disclosure principle used throughout every part above:

1. The raw loop from Part 1, no framework, one mock tool.
2. Swap the mock tool for a real one, add a max-iterations guardrail (Part 8).
3. Add one piece of long-term memory that matters for your actual use case (Part 6).
4. Only then evaluate whether a framework (Part 5) or a second agent (Part 9) earns its complexity for your specific task.

Everything past step 1 is optional, situational, and should be justified by a concrete limitation you've actually hit — not added because a tutorial included it.
