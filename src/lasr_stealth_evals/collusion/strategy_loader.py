import yaml
import os

def load_agent_configs(strategy_name):
    path = os.path.join("strategies", f"{strategy_name}.yaml")

    if not os.path.exists(path):
        print(f"\n🚨 ERROR: Strategy file not found: {path}\n")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data is None:
                print(f"\n🚨 ERROR: Strategy file {path} is empty or invalid YAML.\n")
                return None
            return data
    except Exception as e:
        print(f"\n🚨 ERROR loading strategy: {e}\n")
        return None
