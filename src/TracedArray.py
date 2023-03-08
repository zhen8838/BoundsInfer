from typing import List, Any, Optional, Tuple, Union
import numpy as np
import functools


class MemStorage:
  __loadTraced: List[List[slice]]
  __capacity: int
  __loadedSize: int

  def __init__(self, totalSize: int) -> None:
    self.__capacity = totalSize
    self.__loadedSize = 0
    self.__loadTraced = []

  def TraceLoad(self, idxs: List[slice], reuse=False):
    idxs = list(idxs)
    self.__loadTraced.append(idxs)
    if not reuse:
      self.__loadedSize += functools.reduce(
          lambda acc, s: acc * s,
          [(idx.stop - idx.start) for idx in idxs],
          1)
    assert self.__loadedSize <= self.__capacity, "The LoadSize > Storage Capacity"

  @property
  def LoadedSize(self) -> int:
    return self.__loadedSize

  def Display(self):
    for l in self.__loadTraced:
      print(l)


class MemHierarchy:
  __storage: MemStorage
  __nextStorageCapacity: int
  __totalLoaded: int

  def __init__(self) -> None:
    self.__storage = None
    self.__nextStorageCapacity = 0
    self.__totalLoaded = 0

  def Reset(self):
    self.__totalLoaded = 0

  @property
  def TotalLoaded(self) -> int:
    return self.__totalLoaded

  def Current(self) -> MemStorage:
    return self.__storage

  def __call__(self, nextStorageCapacity) -> Any:
    self.__nextStorageCapacity = nextStorageCapacity
    return self

  def __enter__(self):
    self.__storage = MemStorage(self.__nextStorageCapacity)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.__totalLoaded += self.__storage.LoadedSize
    self.__storage = None
    self.__nextStorageCapacity = 0


GlobalHierarchy = MemHierarchy()


class TarcedArray:
  _array: np.ndarray

  def __init__(self, array: np.ndarray) -> None:
    self._array = array

  def __getitem__(self, idxs: Union[List[slice], Tuple[List[slice], bool]]) -> np.ndarray:
    reuse = False
    if len(idxs) == 2:
      (idxs, reuse) = idxs
    idxs = list(idxs)
    for i in range(len(idxs)):
      if (idxs[i].start == None and idxs[i].stop == None):
        idxs[i] = slice(0, self._array.shape[i])
    GlobalHierarchy.Current().TraceLoad(idxs, reuse=reuse)
    return self._array[idxs[0], idxs[1], idxs[2], idxs[3]]


__all__ = ['GloblMemory', 'TarcedArray', 'MemStorage']

if __name__ == "__main__":
  a = TarcedArray(np.random.randn(4, 5, 6, 7))
  print(a[(slice(0, 1), slice(0, 4), slice(0, 5), slice(0, 3))].shape)
