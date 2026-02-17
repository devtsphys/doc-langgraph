# LangGraph Reference Card & Cheat Sheet

**Complete Guide for Agentic AI Development with Python**

-----

## Table of Contents

1. [Installation & Setup](#installation--setup)
1. [Core Concepts](#core-concepts)
1. [Graph Components](#graph-components)
1. [State Management](#state-management)
1. [Node Functions](#node-functions)
1. [Edge Types](#edge-types)
1. [Conditional Routing](#conditional-routing)
1. [Checkpointing & Persistence](#checkpointing--persistence)
1. [Human-in-the-Loop](#human-in-the-loop)
1. [Complete Method Reference](#complete-method-reference)
1. [Common Patterns](#common-patterns)
1. [Agentic AI Examples](#agentic-ai-examples)

-----

## Installation & Setup

```bash
# Install LangGraph
pip install langgraph

# Install with LangChain
pip install langgraph langchain langchain-openai

# Install with persistence
pip install langgraph[postgres]  # PostgreSQL support
pip install langgraph[sqlite]    # SQLite support
```

### Basic Imports

```python
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
```

-----

## Core Concepts

### 1. **StateGraph**

A graph where nodes are functions that operate on a shared state object.

### 2. **Nodes**

Functions that process and modify the state. Each node receives the current state and returns updates.

### 3. **Edges**

Connections between nodes that determine the flow of execution.

### 4. **State**

A shared data structure passed between nodes, typically a TypedDict or Pydantic model.

### 5. **Checkpointing**

Mechanism to save and restore graph state for persistence and time-travel.

-----

## Graph Components

### Creating a Graph

```python
from langgraph.graph import StateGraph

# Define state schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_step: str
    
# Initialize graph
workflow = StateGraph(AgentState)
```

### Building the Graph

```python
# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("action", action_node)
workflow.add_node("process", process_node)

# Set entry point
workflow.set_entry_point("agent")

# Add edges
workflow.add_edge("agent", "action")
workflow.add_conditional_edges(
    "action",
    should_continue,
    {
        "continue": "process",
        "end": END
    }
)
workflow.add_edge("process", END)

# Compile
app = workflow.compile()
```

-----

## State Management

### TypedDict State (Recommended)

```python
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    user_input: str
    output: str
    iteration_count: int
```

### Pydantic State

```python
from pydantic import BaseModel, Field
from langgraph.graph import add

class State(BaseModel):
    messages: Annotated[list[BaseMessage], add]
    context: dict = Field(default_factory=dict)
    score: float = 0.0
```

### State Reducers

```python
# operator.add - Appends to lists
messages: Annotated[list, operator.add]

# Custom reducer
def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}

metadata: Annotated[dict, merge_dicts]
```

-----

## Node Functions

### Basic Node

```python
def my_node(state: State) -> dict:
    """Process state and return updates"""
    # Access state
    messages = state["messages"]
    
    # Process
    result = process_data(messages)
    
    # Return updates (partial state)
    return {"output": result}
```

### Agent Node with LLM

```python
from langchain_openai import ChatOpenAI

def agent_node(state: State) -> dict:
    llm = ChatOpenAI(model="gpt-4")
    
    messages = state["messages"]
    response = llm.invoke(messages)
    
    return {"messages": [response]}
```

### Tool-Using Node

```python
from langchain_core.tools import tool

@tool
def search_tool(query: str) -> str:
    """Search for information"""
    return f"Results for: {query}"

def tool_node(state: State) -> dict:
    messages = state["messages"]
    last_message = messages[-1]
    
    # Execute tool
    result = search_tool.invoke(last_message.content)
    
    return {"messages": [AIMessage(content=result)]}
```

### Async Node

```python
async def async_node(state: State) -> dict:
    """Async node for concurrent operations"""
    result = await async_api_call()
    return {"output": result}
```

-----

## Edge Types

### 1. Regular Edges

```python
# Direct connection from node A to node B
workflow.add_edge("node_a", "node_b")

# Connect to END
workflow.add_edge("final_node", END)

# Connect from START
workflow.set_entry_point("first_node")
# or
workflow.add_edge(START, "first_node")
```

### 2. Conditional Edges

```python
def route_decision(state: State) -> str:
    """Determine next node based on state"""
    if state["score"] > 0.8:
        return "high_confidence"
    elif state["score"] > 0.5:
        return "medium_confidence"
    else:
        return "low_confidence"

workflow.add_conditional_edges(
    "evaluator",
    route_decision,
    {
        "high_confidence": "approve",
        "medium_confidence": "review",
        "low_confidence": "reject"
    }
)
```

### 3. Multi-Output Conditional Edges

```python
def multi_route(state: State) -> list[str]:
    """Return multiple next nodes"""
    next_nodes = []
    if state["needs_review"]:
        next_nodes.append("review")
    if state["needs_approval"]:
        next_nodes.append("approval")
    return next_nodes or ["end"]
```

-----

## Conditional Routing

### Simple Boolean Router

```python
def should_continue(state: State) -> str:
    """Continue or end based on condition"""
    if state["iteration_count"] < 5:
        return "continue"
    return "end"

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)
```

### Tool Call Router (Prebuilt)

```python
from langgraph.prebuilt import tools_condition, ToolNode

tools = [search_tool, calculator_tool]
tool_node = ToolNode(tools)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Automatically routes based on tool calls
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        END: END
    }
)
```

### Complex State-Based Router

```python
def complex_router(state: State) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check for tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # Check for specific content
    if "FINAL ANSWER" in last_message.content:
        return "end"
    
    # Check iteration limit
    if state.get("iteration_count", 0) > 10:
        return "error"
    
    return "continue"
```

-----

## Checkpointing & Persistence

### Memory Saver (In-Memory)

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# Run with thread
config = {"configurable": {"thread_id": "conversation-1"}}
result = app.invoke({"messages": [HumanMessage("Hello")]}, config)
```

### SQLite Checkpointer

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "user-123"}}
    result = app.invoke(input_data, config)
```

### PostgreSQL Checkpointer

```python
from langgraph.checkpoint.postgres import PostgresSaver

connection_string = "postgresql://user:pass@localhost/dbname"
with PostgresSaver.from_conn_string(connection_string) as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)
```

### Time Travel & State History

```python
# Get state at specific checkpoint
state = app.get_state(config)
print(state.values)  # Current state
print(state.next)    # Next scheduled nodes

# Get state history
for state in app.get_state_history(config):
    print(f"Step: {state.values}")
    print(f"Metadata: {state.metadata}")
```

### Update State

```python
# Update current state
app.update_state(
    config,
    {"messages": [HumanMessage("Updated message")]},
    as_node="agent"  # Optional: specify which node made the update
)
```

-----

## Human-in-the-Loop

### Interrupt Before Node

```python
# Compile with interrupt
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review"]
)

# Run until interrupt
config = {"configurable": {"thread_id": "thread-1"}}
result = app.invoke(input_data, config)

# Get state (will be paused at interrupt)
state = app.get_state(config)
print(f"Paused at: {state.next}")

# Resume after human review
result = app.invoke(None, config)  # Continue from checkpoint
```

### Interrupt After Node

```python
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_after=["decision_node"]
)

# Review and update before continuing
result = app.invoke(input_data, config)
state = app.get_state(config)

# Modify state based on human input
app.update_state(config, {"approved": True})

# Continue
result = app.invoke(None, config)
```

### Manual Approval Pattern

```python
def needs_approval(state: State) -> str:
    if state.get("requires_human_approval"):
        return "await_approval"
    return "continue"

workflow.add_conditional_edges(
    "decision",
    needs_approval,
    {
        "await_approval": "human_review",
        "continue": "proceed"
    }
)

# Compile with interrupt
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review"]
)
```

-----

## Complete Method Reference

### StateGraph Methods

|Method                 |Signature                                                                |Description                             |
|-----------------------|-------------------------------------------------------------------------|----------------------------------------|
|`__init__`             |`StateGraph(state_schema)`                                               |Initialize graph with state schema      |
|`add_node`             |`add_node(name: str, func: Callable)`                                    |Add a node function to the graph        |
|`add_edge`             |`add_edge(start: str, end: str)`                                         |Add direct edge between nodes           |
|`add_conditional_edges`|`add_conditional_edges(source: str, path: Callable, path_map: dict)`     |Add conditional routing                 |
|`set_entry_point`      |`set_entry_point(node: str)`                                             |Set the starting node                   |
|`set_finish_point`     |`set_finish_point(node: str)`                                            |Set the ending node (alternative to END)|
|`compile`              |`compile(checkpointer=None, interrupt_before=None, interrupt_after=None)`|Compile graph into runnable             |

### CompiledGraph Methods

|Method             |Signature                                                      |Description                          |
|-------------------|---------------------------------------------------------------|-------------------------------------|
|`invoke`           |`invoke(input: dict, config: dict = None)`                     |Run graph synchronously              |
|`ainvoke`          |`ainvoke(input: dict, config: dict = None)`                    |Run graph asynchronously             |
|`stream`           |`stream(input: dict, config: dict = None)`                     |Stream graph execution               |
|`astream`          |`astream(input: dict, config: dict = None)`                    |Stream asynchronously                |
|`get_state`        |`get_state(config: dict)`                                      |Get current state of a thread        |
|`get_state_history`|`get_state_history(config: dict)`                              |Get all historical states            |
|`update_state`     |`update_state(config: dict, values: dict, as_node: str = None)`|Update state manually                |
|`get_graph`        |`get_graph()`                                                  |Get graph structure for visualization|

### Checkpointer Methods

|Method|Signature                                            |Description                    |
|------|-----------------------------------------------------|-------------------------------|
|`get` |`get(config: dict)`                                  |Retrieve checkpoint            |
|`put` |`put(config: dict, checkpoint: dict, metadata: dict)`|Save checkpoint                |
|`list`|`list(config: dict)`                                 |List all checkpoints for thread|

### Streaming Methods

|Method      |Signature                            |Description                      |
|------------|-------------------------------------|---------------------------------|
|`stream`    |`stream(input, stream_mode="values")`|Stream with different modes      |
|Stream modes|`"values"`                           |Stream full state after each node|
|            |`"updates"`                          |Stream only state updates        |
|            |`"messages"`                         |Stream only message updates      |
|            |`"events"`                           |Stream all events                |

-----

## Common Patterns

### 1. ReAct Agent Pattern

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Quick ReAct agent
tools = [search_tool, calculator_tool]
model = ChatOpenAI(model="gpt-4")

agent = create_react_agent(model, tools)

# Use it
result = agent.invoke({
    "messages": [HumanMessage("What is 25 * 4 + 10?")]
})
```

### 2. Custom ReAct Loop

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
def agent_node(state: AgentState):
    llm = ChatOpenAI(model="gpt-4").bind_tools(tools)
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: AgentState):
    tool_executor = ToolNode(tools)
    return tool_executor(state)

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

app = workflow.compile()
```

### 3. Multi-Agent Collaboration

```python
class MultiAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    
def researcher_node(state):
    # Research agent
    response = researcher_llm.invoke(state["messages"])
    return {"messages": [response], "next_agent": "writer"}

def writer_node(state):
    # Writing agent
    response = writer_llm.invoke(state["messages"])
    return {"messages": [response], "next_agent": "critic"}

def critic_node(state):
    # Critic agent
    response = critic_llm.invoke(state["messages"])
    if "APPROVED" in response.content:
        return {"messages": [response], "next_agent": "end"}
    return {"messages": [response], "next_agent": "writer"}

def route_agent(state):
    return state["next_agent"]

workflow = StateGraph(MultiAgentState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("researcher")
workflow.add_conditional_edges(
    "researcher",
    route_agent,
    {"writer": "writer"}
)
workflow.add_conditional_edges(
    "writer",
    route_agent,
    {"critic": "critic"}
)
workflow.add_conditional_edges(
    "critic",
    route_agent,
    {"writer": "writer", "end": END}
)

app = workflow.compile()
```

### 4. Planning and Execution

```python
class PlanExecuteState(TypedDict):
    plan: list[str]
    current_step: int
    results: Annotated[list[str], operator.add]
    final_answer: str
    
def planner_node(state):
    # Create plan
    plan = planner_llm.invoke(state["messages"])
    steps = parse_plan(plan.content)
    return {"plan": steps, "current_step": 0}

def executor_node(state):
    # Execute current step
    current_step = state["plan"][state["current_step"]]
    result = executor_llm.invoke(current_step)
    return {
        "results": [result.content],
        "current_step": state["current_step"] + 1
    }

def should_continue(state):
    if state["current_step"] >= len(state["plan"]):
        return "synthesize"
    return "execute"

def synthesizer_node(state):
    # Combine all results
    summary = synthesizer_llm.invoke(
        f"Plan: {state['plan']}\nResults: {state['results']}"
    )
    return {"final_answer": summary.content}

workflow = StateGraph(PlanExecuteState)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_conditional_edges(
    "executor",
    should_continue,
    {"execute": "executor", "synthesize": "synthesizer"}
)
workflow.add_edge("synthesizer", END)

app = workflow.compile()
```

### 5. Reflection Pattern

```python
class ReflectionState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    iterations: int
    
def generate_node(state):
    response = generator_llm.invoke(state["messages"])
    return {
        "messages": [response],
        "iterations": state.get("iterations", 0) + 1
    }

def reflect_node(state):
    reflection = reflector_llm.invoke([
        HumanMessage(f"Review this: {state['messages'][-1].content}")
    ])
    return {"messages": [reflection]}

def should_continue(state):
    if state["iterations"] >= 3:
        return "end"
    if "FINAL" in state["messages"][-1].content:
        return "end"
    return "generate"

workflow = StateGraph(ReflectionState)
workflow.add_node("generate", generate_node)
workflow.add_node("reflect", reflect_node)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "reflect")
workflow.add_conditional_edges(
    "reflect",
    should_continue,
    {"generate": "generate", "end": END}
)

app = workflow.compile()
```

-----

## Agentic AI Examples

### Example 1: Research Agent with Web Search

```python
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI

class ResearchState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    research_query: str
    findings: list[str]
    
search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Search the web for information"""
    return search.run(query)

def research_planner(state):
    """Plan research queries"""
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"Generate 3 research queries for: {state['research_query']}"
    response = llm.invoke([HumanMessage(prompt)])
    return {"messages": [response]}

def researcher(state):
    """Execute searches"""
    llm = ChatOpenAI(model="gpt-4").bind_tools([web_search])
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def synthesizer(state):
    """Synthesize findings"""
    llm = ChatOpenAI(model="gpt-4")
    findings = "\n".join([msg.content for msg in state["messages"]])
    response = llm.invoke([
        HumanMessage(f"Synthesize these research findings:\n{findings}")
    ])
    return {"messages": [response]}

workflow = StateGraph(ResearchState)
workflow.add_node("planner", research_planner)
workflow.add_node("researcher", researcher)
workflow.add_node("tools", ToolNode([web_search]))
workflow.add_node("synthesizer", synthesizer)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_conditional_edges(
    "researcher",
    tools_condition,
    {"tools": "tools", END: "synthesizer"}
)
workflow.add_edge("tools", "researcher")
workflow.add_edge("synthesizer", END)

app = workflow.compile()
```

### Example 2: Code Generation Agent

```python
class CodeGenState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    code: str
    tests_passed: bool
    error_message: str

@tool
def execute_code(code: str) -> str:
    """Execute Python code and return result"""
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return "Success"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def run_tests(code: str, tests: str) -> str:
    """Run tests against code"""
    try:
        exec(code + "\n" + tests)
        return "All tests passed"
    except AssertionError as e:
        return f"Test failed: {str(e)}"

def code_generator(state):
    """Generate code based on requirements"""
    llm = ChatOpenAI(model="gpt-4").bind_tools([execute_code, run_tests])
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def code_reviewer(state):
    """Review and suggest improvements"""
    llm = ChatOpenAI(model="gpt-4")
    last_msg = state["messages"][-1]
    
    if "Error" in last_msg.content:
        return {"messages": [
            AIMessage(content="Code has errors, regenerating...")
        ]}
    return {"messages": [
        AIMessage(content="Code looks good!")
    ]}

def should_continue(state):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    if "Error" in last_msg.content:
        return "generate"
    return "end"

workflow = StateGraph(CodeGenState)
workflow.add_node("generate", code_generator)
workflow.add_node("tools", ToolNode([execute_code, run_tests]))
workflow.add_node("review", code_reviewer)

workflow.set_entry_point("generate")
workflow.add_conditional_edges(
    "generate",
    should_continue,
    {"tools": "tools", "generate": "generate", "end": "review"}
)
workflow.add_edge("tools", "generate")
workflow.add_edge("review", END)

app = workflow.compile()
```

### Example 3: Customer Support Agent with Escalation

```python
class SupportState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    customer_info: dict
    sentiment: str
    escalation_needed: bool
    resolution: str

@tool
def lookup_customer(customer_id: str) -> dict:
    """Look up customer information"""
    # Mock function
    return {"id": customer_id, "tier": "premium", "history": []}

@tool
def create_ticket(issue: str, priority: str) -> str:
    """Create support ticket"""
    return f"Ticket created: {issue} (Priority: {priority})"

def sentiment_analyzer(state):
    """Analyze customer sentiment"""
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"Analyze sentiment (positive/neutral/negative): {state['messages'][-1].content}"
    response = llm.invoke([HumanMessage(prompt)])
    
    sentiment = response.content.lower()
    escalation = "negative" in sentiment or "angry" in sentiment
    
    return {
        "sentiment": sentiment,
        "escalation_needed": escalation
    }

def support_agent(state):
    """Handle customer inquiry"""
    llm = ChatOpenAI(model="gpt-4").bind_tools([
        lookup_customer, create_ticket
    ])
    
    system_msg = "You are a helpful customer support agent."
    if state.get("escalation_needed"):
        system_msg += " This customer needs special attention."
    
    messages = [HumanMessage(system_msg)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def escalation_handler(state):
    """Handle escalated cases"""
    llm = ChatOpenAI(model="gpt-4")
    response = llm.invoke([
        HumanMessage(f"Senior agent handling escalation: {state['messages'][-1].content}")
    ])
    return {"messages": [response]}

def route_support(state):
    if hasattr(state["messages"][-1], "tool_calls") and state["messages"][-1].tool_calls:
        return "tools"
    if state.get("escalation_needed"):
        return "escalate"
    return "end"

workflow = StateGraph(SupportState)
workflow.add_node("analyze_sentiment", sentiment_analyzer)
workflow.add_node("agent", support_agent)
workflow.add_node("tools", ToolNode([lookup_customer, create_ticket]))
workflow.add_node("escalate", escalation_handler)

workflow.set_entry_point("analyze_sentiment")
workflow.add_edge("analyze_sentiment", "agent")
workflow.add_conditional_edges(
    "agent",
    route_support,
    {"tools": "tools", "escalate": "escalate", "end": END}
)
workflow.add_edge("tools", "agent")
workflow.add_edge("escalate", END)

app = workflow.compile()
```

### Example 4: Data Analysis Agent

```python
class AnalysisState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: dict
    analysis_type: str
    insights: list[str]
    visualizations: list[str]

@tool
def load_data(source: str) -> dict:
    """Load data from source"""
    # Mock implementation
    return {"records": 1000, "columns": ["A", "B", "C"]}

@tool  
def statistical_analysis(data: dict, analysis_type: str) -> dict:
    """Perform statistical analysis"""
    return {
        "mean": 50.5,
        "median": 48.0,
        "std": 15.2
    }

@tool
def create_visualization(data: dict, chart_type: str) -> str:
    """Create data visualization"""
    return f"Created {chart_type} chart"

def analyst_node(state):
    """Perform data analysis"""
    llm = ChatOpenAI(model="gpt-4").bind_tools([
        load_data, statistical_analysis, create_visualization
    ])
    
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def insight_generator(state):
    """Generate insights from analysis"""
    llm = ChatOpenAI(model="gpt-4")
    
    findings = "\n".join([
        msg.content for msg in state["messages"]
        if isinstance(msg, AIMessage)
    ])
    
    response = llm.invoke([
        HumanMessage(f"Generate key insights from:\n{findings}")
    ])
    
    return {"messages": [response]}

workflow = StateGraph(AnalysisState)
workflow.add_node("analyst", analyst_node)
workflow.add_node("tools", ToolNode([
    load_data, statistical_analysis, create_visualization
]))
workflow.add_node("insights", insight_generator)

workflow.set_entry_point("analyst")
workflow.add_conditional_edges(
    "analyst",
    tools_condition,
    {"tools": "tools", END: "insights"}
)
workflow.add_edge("tools", "analyst")
workflow.add_edge("insights", END)

app = workflow.compile()
```

### Example 5: Multi-Step Workflow Agent

```python
class WorkflowState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    workflow_steps: list[str]
    current_step: int
    step_results: dict
    final_output: str

def workflow_planner(state):
    """Plan workflow steps"""
    llm = ChatOpenAI(model="gpt-4")
    
    response = llm.invoke([
        HumanMessage(f"Break down this task into steps: {state['messages'][0].content}")
    ])
    
    # Parse steps from response
    steps = [line.strip() for line in response.content.split("\n") if line.strip()]
    
    return {
        "messages": [response],
        "workflow_steps": steps,
        "current_step": 0,
        "step_results": {}
    }

def step_executor(state):
    """Execute current workflow step"""
    llm = ChatOpenAI(model="gpt-4")
    
    current_step = state["workflow_steps"][state["current_step"]]
    previous_results = state.get("step_results", {})
    
    prompt = f"""
    Execute this step: {current_step}
    
    Previous step results: {previous_results}
    """
    
    response = llm.invoke([HumanMessage(prompt)])
    
    # Store result
    new_results = state["step_results"].copy()
    new_results[f"step_{state['current_step']}"] = response.content
    
    return {
        "messages": [response],
        "current_step": state["current_step"] + 1,
        "step_results": new_results
    }

def should_continue_workflow(state):
    if state["current_step"] >= len(state["workflow_steps"]):
        return "finalize"
    return "execute"

def finalizer(state):
    """Combine all step results"""
    llm = ChatOpenAI(model="gpt-4")
    
    summary = "\n".join([
        f"{k}: {v}" for k, v in state["step_results"].items()
    ])
    
    response = llm.invoke([
        HumanMessage(f"Summarize workflow results:\n{summary}")
    ])
    
    return {
        "messages": [response],
        "final_output": response.content
    }

workflow = StateGraph(WorkflowState)
workflow.add_node("planner", workflow_planner)
workflow.add_node("executor", step_executor)
workflow.add_node("finalizer", finalizer)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_conditional_edges(
    "executor",
    should_continue_workflow,
    {"execute": "executor", "finalize": "finalizer"}
)
workflow.add_edge("finalizer", END)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
```

-----

## Advanced Techniques

### 1. Subgraphs

```python
# Create subgraph
subgraph = StateGraph(SubState)
subgraph.add_node("sub_node1", sub_function1)
subgraph.add_node("sub_node2", sub_function2)
subgraph.add_edge("sub_node1", "sub_node2")
subgraph.add_edge("sub_node2", END)
compiled_subgraph = subgraph.compile()

# Use in main graph
main_graph = StateGraph(MainState)
main_graph.add_node("subprocess", compiled_subgraph)
```

### 2. Dynamic Graph Construction

```python
def build_dynamic_graph(tools: list):
    workflow = StateGraph(AgentState)
    
    for i, tool in enumerate(tools):
        workflow.add_node(f"tool_{i}", create_tool_node(tool))
    
    workflow.set_entry_point("tool_0")
    for i in range(len(tools) - 1):
        workflow.add_edge(f"tool_{i}", f"tool_{i+1}")
    
    workflow.add_edge(f"tool_{len(tools)-1}", END)
    return workflow.compile()
```

### 3. Error Handling

```python
def safe_node(state):
    try:
        result = risky_operation(state)
        return {"result": result, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}

def error_handler(state):
    if state.get("error"):
        return "retry"
    return "continue"

workflow.add_conditional_edges(
    "risky_node",
    error_handler,
    {"retry": "risky_node", "continue": "next_node"}
)
```

### 4. Parallel Execution

```python
from langgraph.graph import add

class ParallelState(TypedDict):
    results: Annotated[list, add]

workflow.add_node("parallel_1", func1)
workflow.add_node("parallel_2", func2)
workflow.add_node("parallel_3", func3)
workflow.add_node("combine", combine_results)

# All parallel nodes execute concurrently
workflow.add_edge(START, "parallel_1")
workflow.add_edge(START, "parallel_2")
workflow.add_edge(START, "parallel_3")

# All converge to combine
workflow.add_edge("parallel_1", "combine")
workflow.add_edge("parallel_2", "combine")
workflow.add_edge("parallel_3", "combine")
```

-----

## Debugging & Visualization

### Graph Visualization

```python
from IPython.display import Image, display

# Get graph representation
graph = app.get_graph()

# Draw graph
display(Image(graph.draw_mermaid_png()))

# Or get mermaid code
print(graph.draw_mermaid())
```

### Streaming for Debugging

```python
# Stream state updates
for chunk in app.stream(input_data, stream_mode="updates"):
    print(chunk)

# Stream full state
for state in app.stream(input_data, stream_mode="values"):
    print(f"Current state: {state}")

# Stream events
for event in app.stream(input_data, stream_mode="events"):
    print(f"Event: {event}")
```

### Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("langgraph")

def logged_node(state):
    logger.info(f"Node executing with state: {state}")
    result = process(state)
    logger.info(f"Node result: {result}")
    return result
```

-----

## Best Practices

1. **State Design**: Keep state minimal and well-structured
1. **Node Functions**: Make nodes pure functions when possible
1. **Error Handling**: Always handle exceptions in nodes
1. **Checkpointing**: Use checkpointers for long-running agents
1. **Tool Design**: Create focused, single-purpose tools
1. **Conditional Logic**: Keep routing logic simple and testable
1. **Human-in-Loop**: Use interrupts for critical decisions
1. **Testing**: Test nodes independently before integration
1. **Monitoring**: Use streaming and logging for observability
1. **Documentation**: Document state schema and node purposes

-----

## Quick Reference Commands

```python
# Create graph
workflow = StateGraph(StateSchema)

# Add nodes
workflow.add_node("name", function)

# Add edges
workflow.add_edge("from", "to")
workflow.add_conditional_edges("source", router, mapping)

# Set entry/exit
workflow.set_entry_point("start_node")
workflow.add_edge("end_node", END)

# Compile
app = workflow.compile(checkpointer=MemorySaver())

# Run
result = app.invoke(input_data, config)

# Stream
for chunk in app.stream(input_data):
    print(chunk)

# State management
state = app.get_state(config)
app.update_state(config, updates)
```

-----

## Resources

- **Documentation**: https://langchain-ai.github.io/langgraph/
- **GitHub**: https://github.com/langchain-ai/langgraph
- **Examples**: https://github.com/langchain-ai/langgraph/tree/main/examples
- **Discord**: LangChain Discord community

-----

*Last Updated: February 2026*
*LangGraph Version: 0.2.x+*