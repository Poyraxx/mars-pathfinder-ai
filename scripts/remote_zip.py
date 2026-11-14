"""Bounded, range-validated ZIP reader. Extract only selected AI4Mars members.

Uses the publisher archive directly; no mirrors or whole-archive download fallback.
"""
import io
import urllib.request
import zipfile


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url: str, size: int):
        self.url, self.size, self.position = url, size, 0
        self.cache = {}
        self.block = 2 * 1024 * 1024

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0: raise ValueError("negative offset")
        return self.position

    def read(self, count=-1):
        count = min(self.size-self.position, count if count >= 0 else self.size-self.position)
        if count <= 0: return b""
        end = self.position+count
        pieces = []
        while self.position < end:
            block_id = self.position // self.block
            if block_id not in self.cache:
                start = block_id*self.block
                stop = min(start+self.block,self.size)-1
                req = urllib.request.Request(self.url,headers={"Range":f"bytes={start}-{stop}","User-Agent":"MarsPathfinderResearch/0.1"})
                with urllib.request.urlopen(req,timeout=90) as response:
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
                        raise RuntimeError("Publisher did not honor byte range; refusing a full archive transfer")
                    self.cache[block_id] = response.read(stop-start+1)
                if len(self.cache) > 24:
                    del self.cache[next(iter(self.cache))]
            data = self.cache[block_id]
            off = self.position % self.block
            n = min(len(data)-off,end-self.position)
            if n <= 0: raise IOError("short range response")
            pieces.append(data[off:off+n])
            self.position += n
        return b"".join(pieces)


def open_ai4mars():
    return zipfile.ZipFile(HTTPRangeReader(
        "https://zenodo.org/records/15995036/files/ai4mars-dataset-merged-0.6.zip",
        16232481989))


if __name__ == "__main__":
    archive = open_ai4mars()
    members = archive.namelist()
    print(f"Members: {len(members)}")
    print("\n".join(members[:25]))
    for name in members:
        if name.lower().endswith(("readme.txt","readme.md","label_keys.json")):
            print(name)
            print(archive.read(name).decode("utf-8",errors="replace")[:18000])
