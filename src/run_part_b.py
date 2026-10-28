from src.part_b_customers import build_segments
from src.part_b_lapse import build_lapse_model

if __name__ == "__main__":
    segments = build_segments()
    build_lapse_model(segments)

