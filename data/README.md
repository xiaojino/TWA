## Data Organization
We recommend using the following AzCopy command to download the data, and then put the downloaded files in the folder TWA/data/.

AzCopy executable tools can be downloaded [here](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy).

[TextVQA/Caps/STVQA Data (~62G)](https://tapvqacaption.blob.core.windows.net/data/data).

```
path/to/azcopy copy <folder-link> <target-address> --resursive

# for example, downloading TextVQA/Caps/STVQA Data
path/to/azcopy copy https://tapvqacaption.blob.core.windows.net/data/data <local_path>/TWA/data --recursive

```
