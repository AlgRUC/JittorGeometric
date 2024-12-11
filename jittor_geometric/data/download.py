from __future__ import print_function

import ssl
import os
import os.path as osp
import urllib.request as ur
import zipfile
from six.moves import urllib

from .makedirs import makedirs


GBFACTOR = float(1 << 30)


def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition('/')[2].split('?')[0]
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print('Using exist file', filename)
        return path

    if log:
        print('Downloading', url)

    makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path


def decide_download(url):
    d = ur.urlopen(url)
    size = int(d.info()["Content-Length"])/GBFACTOR

    ### confirm if larger than 1GB
    if size > 1:
        return input("This will download %.2fGB. Will you proceed? (y/N)\n" % (size)).lower() == "y"
    else:
        return True
    

def maybe_log(path, log=True):
    if log:
        print('Extracting', path)


def extract_zip(path, folder, log=True):
    r"""Extracts a zip archive to a specific folder.
    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    if osp.isdir(folder) and os.listdir(folder):
        if log:
            print(f"Folder '{folder}' already contains files, skipping extraction.")
        return
    maybe_log(path, log)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)
    if osp.isdir(folder) and os.listdir(folder):
        if log:
            print(f"Folder '{folder}' already contains files, skipping extraction.")
        return