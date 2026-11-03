# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import glob
import re
import shutil
import sys
from pathlib import Path as PathlibPath
from pathlib import PurePosixPath
from typing import (
    TYPE_CHECKING,
    BinaryIO,
    Generator,
    Literal,
    Optional,
    TextIO,
    Tuple,
    Union,
    overload,
)

from megatron.energon.errors import warn_deprecated

if TYPE_CHECKING:
    import multistorageclient as msc

# A Windows drive path as stored internally, e.g. "/C:/data", which the OS knows as "C:/data".
_WINDOWS_DRIVE_PATH = re.compile(r"^/[A-Za-z]:")


def _resolve_storage_client(profile: str) -> "msc.StorageClient":
    """Resolve the MSC client of a remote profile. MSC is imported only here, so that local
    paths work without it, e.g. on Windows, for which MSC has no build."""
    import multistorageclient as msc

    return msc.resolve_storage_client(f"msc://{profile}")[0]


def _local_os_path(internal_path: str, windows: bool) -> str:
    """Return the OS path of a local internal path; a Windows drive path loses its leading slash."""
    return (
        internal_path[1:] if windows and _WINDOWS_DRIVE_PATH.match(internal_path) else internal_path
    )


class EPath:
    """EPath work much like pathlib.Path but does support local as well as remote file systems.
    To initialize a remote path, prepend the path with "msc://" to use the Multi-Storage Client (MSC).
    For example:

        EPath("msc://profilename/my_datasets/webdataset-000.tar")

    You will need to have your MSC configuration (~/.msc_config.yaml) set up to access the object stores
    or use your rclone configuration. See https://nvidia.github.io/multi-storage-client/config/index.html
    for more information.
    """

    # The path without the protocol. Can also be in S3 for example
    internal_path: PurePosixPath
    # The profile used to access the file system
    profile: str
    # The file system; None for local paths, which use the OS directly
    fs: Optional["msc.StorageClient"]

    def __init__(
        self,
        initial_path: Union[str, "EPath", PathlibPath],
    ) -> None:
        if isinstance(initial_path, EPath):
            self.internal_path = initial_path.internal_path
            self.profile = initial_path.profile
            self.fs = initial_path.fs
        else:
            # Local paths are stored with forward slashes, and Windows drive paths with a leading
            # one, e.g. "/C:/data", so that every path is a PurePosixPath from the root
            if isinstance(initial_path, PathlibPath):
                path = initial_path.absolute().as_posix()
                profile = "default"
            else:
                protocol, profile, path = self._split_protocol(initial_path)
                if protocol is None or protocol == "file":
                    profile = "default"
                    path = PathlibPath(path).absolute().as_posix()
                elif protocol == "rclone":
                    warn_deprecated("rclone:// protocol is deprecated. Use msc:// instead.")
                else:
                    assert protocol == "msc", f"Unknown protocol: {protocol}"
            if not path.startswith("/"):
                path = "/" + path
            self.internal_path = self._resolve(path)
            assert profile is not None
            self.profile = profile
            # Resolve the client. Only depends on the protocol and the first part of the path
            self.fs = None if self.is_local() else _resolve_storage_client(self.profile)

    def __getstate__(self) -> dict:
        return {
            "internal_path": self.internal_path,
            "profile": self.profile,
            # Do not save the fs when serializing, to avoid leaking credentials
        }

    def __setstate__(self, state: dict) -> None:
        self.internal_path = state["internal_path"]
        self.profile = state["profile"]
        self.fs = None if self.is_local() else _resolve_storage_client(self.profile)

    @staticmethod
    def _resolve(path: Union[str, PurePosixPath]) -> PurePosixPath:
        """Resolve a path, removing .. and . components."""
        if isinstance(path, str):
            path = PurePosixPath(path)
        parts = path.parts
        # "//" is the root of a Windows UNC path, e.g. "//server/share"
        if parts[0] not in ("/", "//"):
            raise ValueError("Only absolute paths are supported")
        if ".." in parts or "." in parts:
            new_parts = []
            for part in parts[1:]:
                if part == "..":
                    if len(new_parts) == 0:
                        raise ValueError(f"Path above root: {path}")
                    new_parts.pop()
                elif part == ".":
                    pass
                else:
                    new_parts.append(part)
            path = PurePosixPath(parts[0], *new_parts)
        return path

    @staticmethod
    def _split_protocol(path: str) -> Tuple[Optional[str], Optional[str], str]:
        regex = re.compile(r"^(?P<protocol>[a-z]+)://(?P<profile>[^/]+?)/(?P<path>.+)$")
        m = regex.match(path)
        if m is None:
            return None, None, path
        return m.group("protocol"), m.group("profile"), m.group("path")

    @property
    def _internal_str_path(self) -> str:
        """Return the path as used inside the file system, without the protocol and fs part."""
        return str(self.internal_path)

    @property
    def _local_path(self) -> PathlibPath:
        """Return a local path as the OS knows it."""
        return PathlibPath(_local_os_path(self._internal_str_path, sys.platform == "win32"))

    @overload
    def open(self, mode: Literal["r", "w"] = "r", block_size: Optional[int] = None) -> TextIO: ...

    @overload
    def open(self, mode: Literal["rb", "wb"], block_size: Optional[int] = None) -> BinaryIO: ...

    def open(
        self, mode: Literal["r", "rb", "w", "wb"] = "r", block_size: Optional[int] = None
    ) -> Union[TextIO, BinaryIO]:
        if self.fs is not None:
            return self.fs.open(self._internal_str_path, mode)
        if "w" in mode:
            # Writing creates the parent folders, as it does in object stores
            self._local_path.parent.mkdir(parents=True, exist_ok=True)
        return open(self._local_path, mode, encoding=None if "b" in mode else "utf-8")

    def read_text(self) -> str:
        with self.open() as f:
            return f.read()

    def read_bytes(self) -> bytes:
        with self.open("rb") as f:
            return f.read()

    def write_text(self, text: str) -> None:
        with self.open("w") as f:
            f.write(text)

    def write_bytes(self, data: bytes) -> None:
        with self.open("wb") as f:
            f.write(data)

    def copy(self, target: "EPath") -> None:
        """Copy a file to a new path, possibly between different file systems.

        Args:
            target: The path to the local file to download to.
        """

        if self.is_file():
            if self.fs is None and target.fs is None:
                target._local_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self._local_path, target._local_path)
            elif self.fs == target.fs:
                self.fs.copy(self._internal_str_path, target._internal_str_path)
            elif target.fs is None:
                self.fs.download_file(self._internal_str_path, str(target._local_path))
            elif self.fs is None:
                target.fs.upload_file(target._internal_str_path, str(self._local_path))
            else:
                with self.open("rb") as src_f, target.open("wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
        elif self.fs is None:
            for file_path in self._local_path.rglob("*"):
                if file_path.is_file():
                    EPath(file_path).copy(
                        target / file_path.relative_to(self._local_path).as_posix()
                    )
        else:
            inner_path = EPath(self)
            for fpath in self.fs.list(self._internal_str_path):
                inner_path.internal_path = PurePosixPath("/" + fpath.key)
                inner_path.copy(target / inner_path.relative_to(self))

    @property
    def name(self) -> str:
        return self.internal_path.name

    @property
    def parent(self) -> "EPath":
        new_path = EPath(self)
        new_path.internal_path = self.internal_path.parent
        return new_path

    @property
    def url(self) -> str:
        if self.is_local():
            return str(self._local_path)
        int_path_str = str(self.internal_path)
        return f"msc://{self.profile}{int_path_str}"

    def is_local(self) -> bool:
        return self.profile == "default"

    def is_dir(self) -> bool:
        if self.fs is None:
            return self._local_path.is_dir()
        try:
            return self.fs.info(self._internal_str_path).type == "directory"
        except FileNotFoundError:
            return False

    def is_file(self) -> bool:
        if self.fs is None:
            return self._local_path.is_file()
        return self.fs.is_file(self._internal_str_path)

    def mkdir(self, exist_ok: bool = True, parents: bool = False):
        pass

    def glob(self, pattern) -> Generator["EPath", None, None]:
        search_path = self / pattern
        if self.fs is None:
            # As MSC's POSIX provider globs local files
            for path in glob.glob(str(search_path._local_path), recursive=True):
                yield EPath(PathlibPath(path))
            return
        for path in self.fs.glob(search_path._internal_str_path):
            assert isinstance(path, str)

            new_path = EPath(self)
            new_path.internal_path = self._resolve(self.internal_path / PurePosixPath(path))

            yield new_path

    def size(self) -> int:
        if self.fs is None:
            return self._local_path.stat().st_size
        return self.fs.info(self._internal_str_path).content_length

    def with_suffix(self, suffix: str) -> "EPath":
        new_path = EPath(self)
        new_path.internal_path = self.internal_path.with_suffix(suffix)
        return new_path

    def move(self, target: "EPath") -> None:
        self.copy(target)
        self.unlink()

    def unlink(self) -> None:
        if self.fs is None:
            # Missing files are no error, as in MSC
            return self._local_path.unlink(missing_ok=True)
        return self.fs.delete(self._internal_str_path)

    def relative_to(self, other: "EPath") -> str:
        assert self.profile == other.profile, "Can only use relative_to within same profile"

        return str(self.internal_path.relative_to(other.internal_path))

    def __truediv__(self, other: Union[str, "EPath"]) -> "EPath":
        if isinstance(other, EPath):
            # Always absolute
            return other
        if other.startswith("/") or "://" in other or PathlibPath(other).is_absolute():
            return EPath(other)

        new_path = EPath(self)
        new_path.internal_path = self._resolve(self.internal_path / other)
        return new_path

    def __lt__(self, other: "EPath") -> bool:
        assert self.profile == other.profile, "Cannot compare paths from different profiles"

        return self.internal_path < other.internal_path

    def __str__(self) -> str:
        return self.url

    def __repr__(self) -> str:
        return f"EPath({str(self)!r})"

    def __hash__(self) -> int:
        return hash((self.internal_path, self.profile))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, EPath)
            and self.internal_path == other.internal_path
            and self.profile == other.profile
        )
