import os
import subprocess


class WeightsDownloader:
    @staticmethod
    def download_if_not_exists(url, dest):
        if not os.path.exists(dest):
            WeightsDownloader.download(url, dest)

    @staticmethod
    def download(url, dest):
        print("downloading url: ", url)
        print("downloading to: ", dest)
        subprocess.check_call(["pget", url, dest], close_fds=False)
        subprocess.check_call(["tar", "-zxvf", dest])