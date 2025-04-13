# LlamaIndex Agent Workflow for Data Science

This module provides dynamic workflow planning and execution capabilities for the DataHammer AI Agent Swarm using LlamaIndex's AgentWorkflow.

## Components

### WorkflowManager

Manages LlamaIndex agent workflows for data science tasks, responsible for:
- Creating and configuring task agents
- Building dynamic workflows based on user goals
- Executing workflows and tracking progress
- Handling task outputs and environment updates

### Task Agents

Specialized agents for different data science tasks:

- **DataLoadingTaskAgent**: Loads data from various sources
- **ExplorationTaskAgent**: Explores and understands data
- **CleaningTaskAgent**: Cleans and preprocesses data
- **AnalysisTaskAgent**: Performs in-depth data analysis
- **ModelingTaskAgent**: Builds and evaluates predictive models
- **VisualizationTaskAgent**: Creates data visualizations
- **ReportingTaskAgent**: Generates reports and documentation

## Usage

The LlamaIndex workflow module is used by the OrchestratorAgent to dynamically plan and execute workflows based on user goals and data characteristics.

```python
# Initialize the workflow manager
workflow_manager = WorkflowManager()

# Register task agents
workflow_manager.register_default_agents()

# Create a workflow
workflow = workflow_manager.create_workflow(goals, environment)

# Execute the workflow
updated_env = workflow_manager.execute_workflow(environment)
```

## Workflow Planning Process

1. The WorkflowManager analyzes the user goals and available agents
2. It creates a directed acyclic graph (DAG) of workflow tasks
3. Each task is assigned to a specialized agent
4. Dependencies between tasks are established

## Workflow Execution Process

1. The AgentWorkflow executes tasks in the correct order (respecting dependencies)
2. As tasks complete, their results are added to the environment
3. The workflow continues until all tasks are completed

## Integration with LlamaIndex

This workflow system directly uses LlamaIndex's AgentWorkflow, which provides:
- Task-based workflow execution
- Dependency management between tasks
- Parallel execution of independent tasks
- Specialized agent roles for different tasks