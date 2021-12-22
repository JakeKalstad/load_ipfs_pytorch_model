import errno
import os
import sys
import torch
import io
import errno
import hashlib
import os
import shutil
import sys
import tempfile
import torch
import requests
import tarfile


def download_cid_to_file(url, cid, dst, hash_prefix=None):
    r"""Download object at the given CID to a local path.

    Args:
        url (string): URL of the IPFS instance
        cid (string): CID of the model to download
        dst (string): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True

    Example:
        >>> torch.hub.download_url_to_file('the-models-ipfs-cid-here', '/tmp/temporary_file')

    """
    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
    response = requests.post(url+"/get?arg="+cid)
    contents = response.content
    tar = tarfile.open(fileobj=io.BytesIO(contents))
    for member in tar.getmembers():
        if member.isfile:
            extractedFile = tar.extractfile(member)
            if extractedFile is not None:
                f.write(extractedFile.read())
    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)


def load_state_dict_from_ipfs(cid, model_dir=None, url="http://127.0.0.1:5001/api/v0", map_location=None, check_hash=False, file_name=None):
    r"""Loads the Torch serialized object at the given IPFS CID.

    If downloaded file is a zip file, it will be automatically
    decompressed.

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        cid (string): CID of the model to download
        url (string): URL of the IPFS instance
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (string, optional): name for the downloaded file. Filename from ``url`` will be used if not set.

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_ipfs('my-cid-goes-here')

    """
    if model_dir is None:
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise
    filename = cid
    if file_name is not None:
        filename = file_name

    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = torch.hub.HASH_REGEX.search(filename)
            hash_prefix = r.group(1) if r else None
        download_cid_to_file(url, cid, cached_file, hash_prefix)

    if torch.hub._is_legacy_zip_format(cached_file):
        return torch.hub._legacy_zip_load(cached_file, model_dir, map_location)
    return torch.load(cached_file, map_location=map_location)
