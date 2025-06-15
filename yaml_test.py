import sys
import json
from plan_parser import load_training_plan  # Ensure this module is on PYTHONPATH

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_plan.py <training_plan.yaml>")
        sys.exit(1)

    plan_path = sys.argv[1]

    # Load training plan
    training_plan = load_training_plan(plan_path)

    # Simulate daemon output with dummy values
    dummy_values = json.dumps([{"values": [f"dummy_value_{i}" for i in range(10)]}, {"status": 0}])

    # Render and print prompts
    for idx, task in enumerate(training_plan.tasks):
        rendered = task.render_prompt(dummy_values)
        print(f"\n--- Task {idx + 1} Prompt ---")
        print(rendered)

    print(training_plan.data['properties']['generations'])

if __name__ == "__main__":
    main()
