"""
db/ — SQLite results database for the adversarial testing suite.

Modules
-------
schema   — table definitions, init_db(), get_connection()
writer   — ResultsWriter context manager used by all experiment scripts

Quick start::

    from adversarial_suite.db.writer import ResultsWriter

    with ResultsWriter("my_experiment") as w:
        model_id  = w.ensure_model("Qwen/Qwen2.5-7B", registry=reg)
        prompt_id = w.ensure_prompt("Hello, world", complexity="simple")
        w.add_result(
            model_id=model_id,
            prompt_id=prompt_id,
            attack_type="honest",
            token_match_rate=1.0,
            pass_fail=True,
        )
"""

from adversarial_suite.db.schema import init_db, get_connection, DEFAULT_DB_PATH
from adversarial_suite.db.writer import ResultsWriter

__all__ = ["init_db", "get_connection", "DEFAULT_DB_PATH", "ResultsWriter"]
