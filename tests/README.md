# BPE-Transformer: Pytest

The tests inside this directory `test` were cloned from [this repo](https://github.com/damek/cs336-assignment1-basics/tree/main/tests). 



To run all tests:

`uv run pytest`

To run a specific test file only:

`uv run pytest test/test_tokenizer.py`

To run a specific function from a specific file:

`uv run pytest tests/test_tokenizer.py::test_roundtrip_unicode_string_with_special_tokens`

Add `-v` or `--v` in the command line to debug errors.