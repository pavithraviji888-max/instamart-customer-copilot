from pathlib import Path
import pandas as pd

for p in ["data/raw","data/clean","data/policies","outputs","state","sql"]:
    Path(p).mkdir(parents=True,exist_ok=True)

from src.part_a_data import build_part_a
from src.part_a_validate import validate
from src.part_b_customers import build_segments
from src.part_b_lapse import build_lapse_model
from src.part_c_triage import run_part_c
from src.part_d_rag import evaluate
from src.part_d_agent import run_part_d

def main():
    build_part_a()
    validate()
    segments=build_segments()
    build_lapse_model(segments)
    run_part_c()
    evaluate()
    run_part_d()
    print("Pipeline finished. Review outputs and README before submission.")

if __name__ == "__main__":
    main()
