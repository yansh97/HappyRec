from os import PathLike
from pathlib import Path
from typing import IO

import requests
from rich.progress import Progress


class IOHandle:
    """IOHandle is a wrapper to support filepaths and IO objects simultaneously.

    IOHandle can be used as a context manager in the ``with`` statements. When exiting
    the context, the IO object will be closed automatically if it is opened by the
    function :py:func:`happyrec.utils.fileio.get_io_handle`.
    """

    def __init__(self, io: IO, shoule_close: bool) -> None:
        """Initialize the IOHandle with an IO object and a flag indicating whether
        the IO object should be closed when exiting the context.

        :param io: The IO object.
        :param shoule_close: A flag indicating whether the IO object should be closed
            when exiting the context.
        """
        self.io = io
        self._should_close = shoule_close

    def __enter__(self) -> "IOHandle":
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.io.flush()
        if self._should_close:
            self.io.close()


def get_io_handle(file: str | PathLike | IO, mode: str | None = None) -> IOHandle:
    """Get an IOHandle object from a file path or an IO object.

    If the input is a file path, the function will open the file and return an
    IOHandle object which will close the file when exiting the context. If the input
    is an IO object, the function will return an IOHandle object which do nothing
    when exiting the context.

    :param file: The file path or an IO object.
    :param mode: The mode to open the file. If the input is an IO object, this
        parameter will be ignored. Default: ``None``.
    :raises TypeError: If the input is neither a file path nor an IO object.
    :raises ValueError: If the input is a file path but the mode is not specified.
    :return: An IOHandle object.
    """
    if isinstance(file, IO):
        return IOHandle(file, False)
    if isinstance(file, str | PathLike):
        if mode is None:
            raise ValueError("The mode cannot be None when the input is a file path.")
        encoding = None if "b" in mode else "utf-8"
        return IOHandle(open(file, mode=mode, encoding=encoding), True)
    raise TypeError("The input must be a file path or an IO object.")


def download(url: str, path: str | PathLike) -> None:
    """Download a file from a URL.

    :param url: The URL of the file.
    :param path: The path to save the file.
    """
    with requests.get(url, stream=True, timeout=10) as resp:
        with get_io_handle(path, "wb") as out:
            content_length = resp.headers.get("Content-Length")
            total_length = int(content_length) if content_length else None
            with Progress() as progress:
                task = progress.add_task(
                    f"Downloading {Path(path).name} ...", total=total_length
                )
                for chunk in resp.iter_content(chunk_size=1024):
                    if not chunk:
                        break
                    progress.update(task, advance=len(chunk))
                    out.io.write(chunk)
