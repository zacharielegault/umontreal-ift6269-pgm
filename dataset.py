import logging
import os
import urllib.request
from abc import abstractmethod, ABC
from typing import Dict, Iterable, Union

logger = logging.getLogger(__name__)



class DownloadableDataset(GeneExpressionDataset, ABC):
    """Sub-class of ``GeneExpressionDataset`` which downloads its data to disk and
    then populates its attributes with it.

    In particular, it has a ``delayed_populating`` parameter allowing for instantiation
    without populating the attributes.

    :param urls: single or multiple url.s from which to download the data.
    :param filenames: filenames for the downloaded data.
    :param save_path: path to data storage.
    :param delayed_populating: If False, populate object upon instantiation.
        Else, allow for a delayed manual call to ``populate`` method.
    """

    def __init__(
        self,
        urls: Union[str, Iterable[str]] = None,
        filenames: Union[str, Iterable[str]] = None,
        save_path: str = "data/",
        delayed_populating: bool = False,
    ):
        super().__init__()
        if isinstance(urls, str):
            self.urls = [urls]
        elif urls is None:
            self.urls = []
        else:
            self.urls = urls
        if isinstance(filenames, str):
            self.filenames = [filenames]
        elif filenames is None:
            self.filenames = [
                "dataset_{i}".format(i=i) for i, _ in enumerate(self.urls)
            ]
        else:
            self.filenames = filenames

        self.save_path = os.path.abspath(save_path)
        self.download()
        if not delayed_populating:
            self.populate()

    def download(self):
        for url, download_name in zip(self.urls, self.filenames):
            _download(url, self.save_path, download_name)

    @abstractmethod
    def populate(self):
        """Populates a ``DonwloadableDataset`` object's data attributes.

        E.g by calling one of ``GeneExpressionDataset``'s ``populate_from...`` methods.
        """
        pass


def _download(url: str, save_path: str, filename: str):
    """Writes data from url to file."""
    if os.path.exists(os.path.join(save_path, filename)):
        logger.info("File %s already downloaded" % (os.path.join(save_path, filename)))
        return

    r = urllib.request.urlopen(url)
    logger.info("Downloading file at %s" % os.path.join(save_path, filename))

    def read_iter(file, block_size=1000):
        """Given a file 'file', returns an iterator that returns bytes of
        size 'blocksize' from the file, using read()."""
        while True:
            block = file.read(block_size)
            if not block:
                break
            yield block

    # Create the path to save the data
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, filename), "wb") as f:
        for data in read_iter(r):
            f.write(data)
