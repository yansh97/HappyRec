import hashlib
import lzma
from os import PathLike
from pathlib import Path

import requests
from rich.progress import Progress

from .logger import logger


def compress(src: str | PathLike, dst: str | PathLike) -> None:
    """Compress a file to a .xz file.

    :param src: The path of the file to be compressed.
    :param dst: The path of the compressed file.
    """
    src = Path(src)
    dst = Path(dst)
    logger.info(f"Compressing {src.name} to {dst.name} ...")
    compressor = lzma.LZMACompressor()
    with open(src, "rb") as f_in:
        with open(dst, "wb") as f_out:
            with Progress() as progress:
                task = progress.add_task(
                    f"Compressing {src.name} to {dst.name} ...",
                    total=src.stat().st_size,
                )
                while chunk := f_in.read(1024):
                    progress.update(task, advance=len(chunk))
                    f_out.write(compressor.compress(chunk))
                f_out.write(compressor.flush())


def decompress(src: str | PathLike, dst: str | PathLike) -> None:
    """Decompress a .xz file.

    :param src: The path of the compressed file.
    :param dst: The path of the decompressed file.
    """
    src = Path(src)
    dst = Path(dst)
    logger.info(f"Decompressing {src.name} to {dst.name} ...")
    decompressor = lzma.LZMADecompressor()
    with open(src, "rb") as f_in:
        with open(dst, "wb") as f_out:
            with Progress() as progress:
                task = progress.add_task(
                    f"Decompressing {src.name} to {dst.name} ...",
                    total=src.stat().st_size,
                )
                while chunk := f_in.read(1024):
                    progress.update(task, advance=len(chunk))
                    f_out.write(decompressor.decompress(chunk))


def checksum(path: str | PathLike) -> str:
    """Calculate the SHA256 checksum of a file.

    :param path: The path of the file.
    :return: The SHA256 checksum of the file.
    """
    path = Path(path)
    logger.info(f"Calculating the checksum of {path.name} ...")
    sha256 = hashlib.sha256()
    with open(path, "rb") as f_in:
        with Progress() as progress:
            task = progress.add_task(
                f"Calculating checksum of {path.name} ...", total=path.stat().st_size
            )
            while chunk := f_in.read(1024):
                progress.update(task, advance=len(chunk))
                sha256.update(chunk)
    return sha256.hexdigest()


def download(url: str, dst: str | PathLike) -> None:
    """Download a file from a URL.

    :param url: The URL of the file.
    :param dst: The path of the downloaded file.
    """
    dst = Path(dst)
    logger.info(f"Downloading {url} to {dst.name} ...")
    with requests.get(url, stream=True, timeout=10) as resp:
        with open(dst, "wb") as f_out:
            content_length = resp.headers.get("Content-Length")
            size = int(content_length) if content_length else None
            with Progress() as progress:
                task = progress.add_task(f"Downloading {dst.name} ...", total=size)
                for chunk in resp.iter_content(chunk_size=1024):
                    if not chunk:
                        break
                    progress.update(task, advance=len(chunk))
                    f_out.write(chunk)
