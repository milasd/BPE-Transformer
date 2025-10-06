ENCODING_STD = "utf-8"

# Pre-tokenization:
# Regex-based pattern (used by GPT-2; Radford et al., 2019) from github.com/openai/tiktoken/pull/234/files
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
