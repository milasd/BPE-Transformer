import os
import regex as re
from pathlib import Path
from typing import BinaryIO
from multiprocessing import Pool, cpu_count
from collections import Counter
from functools import reduce

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


# Pre-tokenization:
# Regex-based pattern (used by GPT-2; Radford et al., 2019) from github.com/openai/tiktoken/pull/234/files
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
ENCODING_STD = "utf-8"


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(file_path: Path, start: int, end: int) -> Counter[bytes, int]:
    """
    Apply pretokenization to a chunk with defined start and end positions in a file.

    Args:
        file_path: Path of file to load the chunk
        start: Index of chunk start
        end: Index of chunk end

    Returns:
        Counter dict with (pretoken, n. of occurrences) in the chunk.
    """
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode(ENCODING_STD, errors="ignore")
        counter: Counter[tuple[bytes], int] = Counter()
        # Find matching patterns for pretokens in chunk
        for match in re.finditer(PAT, chunk):
            # bytes tuple version of the pretoken
            b = tuple(match.group().encode(ENCODING_STD))
            counter[b] += 1
        return counter


def parallel_pretokenization(
    file_path: Path, num_processes: int = None, split_token: bytes = b"<|endoftext|>"
) -> Counter[bytes, int]:
    """
    Pallelized pretokenization using multiprocessing.
    TODO: add support to multiple split tokens.

    Args:
        file_path: Path to the input file
        num_processes: Number of processes to use (defaults to CPU count)
        split_token: Token to use for chunk boundaries

    Returns:
        Counter object with combined token counts from all chunks
    """
    if num_processes is None or num_processes == 0:
        num_processes = cpu_count()

    # Get chunk boundaries
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)

    # Prepare arguments for parallel processes
    chunk_args = [(file_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

    # Pretokenize chunks in parallel
    with Pool(processes=num_processes) as pool:
        chunk_pretoken_counters = pool.starmap(pretokenize_chunk, chunk_args)

    # Combine all chunks pretoken counters
    pretokens_counter = reduce(lambda x, y: x + y, chunk_pretoken_counters, Counter())

    return pretokens_counter


def serial_pretokenization(file_path: Path, split_token: bytes = b"<|endoftext|>") -> Counter[bytes, int]:
    """
    Original serial implementation for pre-tokenization in text chunks.
    TODO: add support to multiple split tokens.

    Args:
        file_path: Path to the input file
        num_processes: Number of processes to use (defaults to CPU count)
        split_token: Token to use for chunk boundaries

    Returns:
        Counter object with combined token counts from all chunks
    """
    with open(file_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, split_token)

        pretokens_counter = Counter()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode(ENCODING_STD)

            for match in re.finditer(PAT, chunk):
                bytes_list = tuple(match.group().encode(ENCODING_STD))
                pretokens_counter[bytes_list] += 1

    return pretokens_counter


def pretokenize(
    file_path: Path,
    split_token: bytes = b"<|endoftext|>",
    parallel_processing: bool | None = True,
    n_workers: int | None = 4,
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

    Returns:
        Counter object with combined token counts from all chunks
    """
    return (
        parallel_pretokenization(file_path=file_path, split_token=split_token, num_processes=n_workers)
        if parallel_processing
        else serial_pretokenization(file_path=file_path, split_token=split_token)
    )
