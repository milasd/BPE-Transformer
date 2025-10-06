import os
import regex as re
from pathlib import Path
from typing import BinaryIO
from multiprocessing import Pool, cpu_count
from collections import Counter
from functools import reduce

from bpe_transformer.tokenization.settings import ENCODING_STD, PAT

"""
Apply pre-tokenization (following GPT-2 pattern) to an input file.

The function "find_chunk_boundaries" implementation was obtained from:


Copyright statement required for usage below:

Copyright 2025 Stanford University

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), 
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software 
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, 
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
DAMAGES  OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


def pretokenize(
    file_path: Path,
    training: bool | None = True,
    parallel_processing: bool | None = True,
    n_workers: int | None = 4,
    special_tokens: list[str] = None,
) -> Counter[bytes, int]:
    """
    Apply pre-tokenization in text chunks.
    Processes data parallelly by default. If n. of workers is not provided, will use
    TODO: add support to multiple split tokens.

    Args:
        file_path: Path to the input file
        num_processes: Number of processes to use (defaults to CPU count)
        split_token: Token to use for chunk boundaries
        parallel_processing: If true, will parallelize the pretokenization processing.
        n_workers: Number of workers for parallel processing if it's selected; Default: 4
        special_tokens: List of special tokens to keep intact

    Returns:
        Counter object with combined token counts from all chunks
    """
    return (
        parallel_pretokenization(
            file_path=file_path, training=training, num_processes=n_workers, special_tokens=special_tokens
        )
        if parallel_processing
        else serial_pretokenization(file_path=file_path, training=training, special_tokens=special_tokens)
    )


def parallel_pretokenization(
    file_path: Path, num_processes: int = None, training: bool | None = True, special_tokens: list[str] = None
) -> Counter[bytes, int]:
    """
    Pallelized pretokenization using multiprocessing.
    TODO: add support to multiple split tokens.

    Args:
        file_path: Path to the input file
        num_processes: Number of processes to use (defaults to CPU count)
        split_token: Token to use for chunk boundaries
        special_tokens: List of special tokens to keep intact

    Returns:
        Counter object with combined token counts from all chunks
    """
    if num_processes is None or num_processes == 0:
        num_processes = 4

    if num_processes > cpu_count():
        num_processes = cpu_count()

    # Get chunk boundaries
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens)

    # Prepare arguments for parallel processes
    chunk_args = [
        (file_path, start, end, training, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    # Pretokenize chunks in parallel
    with Pool(processes=num_processes) as pool:
        chunk_pretoken_counters = pool.starmap(pretokenize_chunk, chunk_args)

    # Combine all chunks pretoken counters
    pretokens_counter = reduce(lambda x, y: x + y, chunk_pretoken_counters, Counter())

    return pretokens_counter


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    training: bool | None = True,
    special_tokens: list[str] = None,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    # Convert special tokens to bytes for searching
    if special_tokens:
        split_tokens_bytes = [token.encode(ENCODING_STD) for token in special_tokens]
    else:
        split_tokens_bytes = [b"\n"]  # Default to newline if no special tokens

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find any of the special tokens in the mini chunk
            earliest_pos = -1
            for split_token in split_tokens_bytes:
                found_at = mini_chunk.find(split_token)
                if found_at != -1:
                    if earliest_pos == -1 or found_at < earliest_pos:
                        earliest_pos = found_at

            if earliest_pos != -1:
                chunk_boundaries[bi] = initial_position + earliest_pos
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(
    file_path: Path, start: int, end: int, training: bool | None = True, special_tokens: list[str] = None
) -> Counter[bytes, int]:
    """
    Apply pretokenization to a chunk with defined start and end positions in a file.

    Args:
        file_path: Path of file to load the chunk
        start: Index of chunk start
        end: Index of chunk end
        training: boolean defining if the pre-tokens are for training or not.
                  if it's for training, special tokens will not be included in text parts.
                  else, special tokens will be added as separate elements as they are, and never pretokenized.
        special_tokens: List of special tokens to keep intact

    Returns:
        Counter dict with (pretoken, n. of occurrences) in the chunk.
    """
    if special_tokens is None:
        special_tokens = []

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode(ENCODING_STD, errors="ignore")
        counter: Counter[tuple[bytes], int] = Counter()

        # Split on special tokens to prevent merging across document boundaries
        text_parts = split_on_special_tokens(text=chunk, training=training, special_tokens=special_tokens)

        # Pretokenize each part separately
        for part in text_parts:
            if not part:  # Skip empty parts
                continue
            # Use pretokenize_text helper
            pretokens = pretokenize_text(part)
            for pretoken in pretokens:
                counter[tuple(pretoken)] += 1
        return counter


def split_on_special_tokens(text: str, training: bool | None = True, special_tokens: list[str] = None) -> list[str]:
    """
    Split text on special tokens to prevent merging across boundaries.

    Args:
        text: Text string to split
        training: boolean defining if the pre-tokens are for training or not.
                  if it's for training, special tokens will not be included in text parts.
                  else, special tokens will be added as separate elements as they are, and never pretokenized.
        special_tokens: List of special tokens to split on

    Returns:
        List of text parts
    """
    if special_tokens:
        # Create regex pattern to split on special tokens
        # Sort it to take care of possible "overlapping/double" special tokens
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        escaped_specials = [re.escape(token) for token in sorted_tokens]
        # pretokenization for training: remove special tokens, else add them.
        split_pattern = "|".join(escaped_specials) if training else f"({'|'.join(escaped_specials)})"
        text_parts = re.split(split_pattern, text)
    else:
        text_parts = [text]
    return text_parts


def pretokenize_text(text: str) -> list[bytes]:
    """
    Apply GPT-2 pretokenization pattern to a text string.

    Args:
        text: Text string to pretokenize

    Returns:
        List of pretokens as bytes
    """
    pretokens = []
    for match in re.finditer(PAT, text):
        pretoken_bytes = match.group().encode(ENCODING_STD)
        pretokens.append(pretoken_bytes)
    return pretokens


def serial_pretokenization(
    file_path: Path, training: bool | None = True, special_tokens: list[str] = None
) -> Counter[bytes, int]:
    """
    Original serial implementation for pre-tokenization in text chunks.
    TODO: add support to multiple split tokens.

    Args:
        file_path: Path to the input file
        training: boolean defining if the pre-tokens are for training or not.
                  if it's for training, special tokens will not be included in text parts.
                  else, special tokens will be added as separate elements as they are, and never pretokenized.
        num_processes: Number of processes to use (defaults to CPU count)
        split_token: Token to use for chunk boundaries
        special_tokens: List of special tokens to keep intact

    Returns:
        Counter object with combined token counts from all chunks
    """
    if special_tokens is None:
        special_tokens = []

    with open(file_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens)

        pretokens_counter = Counter()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode(ENCODING_STD)

            # Split on special tokens to prevent merging across document boundaries
            text_parts = split_on_special_tokens(text=chunk, training=training, special_tokens=special_tokens)

            # Pretokenize each part separately
            for part in text_parts:
                if not part:  # Skip empty parts
                    continue
                # Use pretokenize_text helper
                pretokens = pretokenize_text(part)
                for pretoken in pretokens:
                    pretokens_counter[tuple(pretoken)] += 1

    return pretokens_counter
