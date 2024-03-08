from typing import List, Optional
from affinitree import AffTree, AffFunc


def from_sequential(layers: List[str], aff_funcs: List['AffFunc'], precondition: Optional['AffTree'], csv: Optional[str]) -> 'AffTree':
    ...

def read_npz(dim: int, filename: str, precondition: Optional['AffTree'], csv: Optional[str]) -> 'AffTree':
    ...