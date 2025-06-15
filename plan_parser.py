import os
import yaml
import json
from jinja2 import Template
from typing import Any, Dict, List


class Task:
    def __init__(self, definition_path: str):
        self.definition_path = definition_path
        self.data = self._load_yaml(definition_path)
        self.setup = self.data['setup_recipe']
        self.verify = self.data['verification_recipe']

    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def render_prompt(self, daemon_output: str) -> str:
        report_data = json.loads(daemon_output)
        value_list = report_data[0].get("values", [])

        context = {
            var["name"]: value_list[var["index"]]
            for var in self.data.get("variables", [])
        }

        template = Template(self.data["prompt_template"])
        return template.render(**context)


class TrainingPlan:
    def __init__(self, plan_path: str):
        self.plan_path = plan_path
        self.data = self._load_yaml(plan_path)
        self.pop_size = self.data['properties']['pop_size']
        self.concurrent = self.data['properties']['concurrent']
        self.generations = self.data['properties']['generations']
        self.sigma = self.data['properties']['sigma']
        self.lr_initial = self.data['properties']['lr_initial']
        self.lr_min = self.data['properties']['lr_min']
        self.evaluation_interval = self.data['properties']['evaluation_interval']
        self.improvement_threshold = self.data['properties']['improvement_threshold']
        self.max_steps = self.data['properties']['max_steps']
        self.tasks = [Task(task_info["definition"]) for task_info in self.data.get("tasks", [])]

    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)


def load_training_plan(plan_path: str) -> TrainingPlan:
    """
    Load a training plan and return a TrainingPlan object.

    Args:
        plan_path (str): Path to the YAML training plan file.

    Returns:
        TrainingPlan: Parsed training plan as a Python-native object.
    """
    return TrainingPlan(plan_path)

