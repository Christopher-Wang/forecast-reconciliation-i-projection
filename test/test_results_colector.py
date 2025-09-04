import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, indicator="pyproject.toml", pythonpath=True, cwd=True)

from src.experiments.results_collector import collect_tiny_results

if __name__ == "__main__":
    collect_tiny_results()
