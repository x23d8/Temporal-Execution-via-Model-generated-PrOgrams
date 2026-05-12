"""Public API for src.prompts.

Phase 1 (duration / date_arith)
    from src.prompts.templates   import get_template, build_messages
    from src.prompts.shot_pools  import get_shots, SHOT_POOLS

Phase 2 (arithmetic MCQ)
    from src.prompts.mcq_templates  import build_mcq_messages
    from src.prompts.mcq_shot_pools import load_mcq_shots, get_mcq_shots
"""

# Phase 1
from .templates   import get_template, build_messages, TEMPLATES        # noqa: F401
from .shot_pools  import get_shots, SHOT_POOLS                          # noqa: F401

# Phase 2 — arithmetic MCQ (direct)
from .mcq_templates  import (                                            # noqa: F401
    build_mcq_messages, MCQ_SYSTEM,
    build_compute_messages, build_match_messages,
    extract_computed_answer, extract_letter,
)
from .mcq_shot_pools import load_mcq_shots, get_mcq_shots               # noqa: F401
from .mcq_executor   import parse_params, execute, detect_category      # noqa: F401
