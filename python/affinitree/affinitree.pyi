
from typing import List, Tuple, Any, Sequence, Set, Optional
import numpy.typing as npt


class AffTree:
    """
    A class used to represent pice-wise linear functions (i.e., piece-wise linear neural networks) in the form of a
    decision tree.

    Each AffTree represents a piece-wise linear function g : R^m -> R^n, where m is called the input dimension and n is called
    the output dimension.

    Nodes in this Tree are of type AffNode.

    Non-terminal nodes in this tree represent the boundaries of the different regions, terminal nodes represent the linear function
    of a given linear region.

    While the primary function of AffTrees is to represent neural networks in a white-box fashion, they can efficiently handle any
    piece-wise linear function, even those that are not continuous.

    NOTE:
        Vectors are represented as numpy arrays with one axis
        Matrices are represented as numpy arrays with two axes

    Constructors
    -----------
    identity(dim_in:int) -> AffTree
        Constructs an AffTree that represents the identity function id: R^{dim_in} -> R^{dim_in} with id(x)=x.

    from_array(weights:npt.NDArray (Matrix), bias: npt.NDArray (Vector))
        Constructs an AffTree that represents the function f(x)=weights*x+bias.

    Methods
    -----------
    apply_function(aff_func: AffFunc)
        Applies aff_func to the AffTree. If a tree represented the function g, it represents the function f(x) = aff_func(g(x)) after execution-

    compose(other:AffTree)
        Composes two AffTrees. If self represented the function g and other represents the function f, self represents f(g(x)) after execution.

    evaluate(npt.NDArray (Vector))->npt.NDArray (Vector)
        Evaluates the tree on a given input vector

    infeasible_elimination(precondition: Optional[Polytope])
        Removes all paths from the AffTree that cannot be reached by any input that satisfies the precondition. If no precondition
        is specified, all inputs are considered.

    size()-> int
        Returns the total number of nodes in the tree.

    num_leaves() -> int
        Returns the number of leaf nodes in the tree.

    parent(node:AffNode)->AffNode
        Returns the parent node of node

    child(node:AffNode, label:int)-> AffNode
        Returns the child node that is reached from the input node if (label=1) the condition of node is satisfied or (label=0) unsatisfied.

    nodes() -> list[AffNodes]
        Returns a list of all nodes in the tree.

    terminals() -> list[AffNodes]
        Returns a list of all terminal nodes in the tree.

    dfs() -> List[Tuple[int, 'AffNode']]
        Returns the nodes of the tree, numbered by a depth-first search

    edges() -> List[Tuple[AffNode, int, AffNode]]
        Returns the edges of the tree. Each tuple is of the form (source, label,target)

    polyhedra() -> List[Tuple[int, 'AffNode', 'Polytope']]
        Returns the linear regions of the AffTree. Each tuple is of the form (num, Terminal, Polytope). The Polytope
        characterizes the linear region and Terminal contains the AffineFunction that is associated with that linear region.

    """
    
    root: AffNode
    
    @classmethod
    def identity(cls, dim: int) -> 'AffTree':
        """
        Inputs:
            dim : int, the input dimension of this AffTree.

        Constructs a simple AffTree representing the identity function in dim dimensions.
        Evaluating this tree yields f(x)=x.
        """
        
    @classmethod
    def from_aff(cls, func: AffFunc) -> 'AffTree':
        pass
    
    @classmethod
    def from_array(cls, weights: npt.NDArray, bias: npt.NDArray) -> 'AffTree':
        """
        Inputs:
            weights : npt.NDArray (Matrix n x m), weight Matrix
            bias    : npt.NDArray (Vector n), bias vector

        Constructs a simple AffTree representing the function f(x)=weights*x + bias.

        Assumptions:
            Ensure that the dimensions of the weights and bias are compatible, i.e., weights.shape[0]=bias.shape[0]
        """
    
    @classmethod
    def from_poly(cls, precondition: Polytope, func: AffFunc) -> AffTree:
        pass
        
    def apply_func(aff_func: 'AffFunc'):
        """
        Inputs:
            aff_func: AffFunc, affine function that is to be applied.

        "Applies" aff_func to the AffTree. Specifically: Assume that the current tree represents a function g. Then,
        after the apply_func call, the tree represents the function f(x)=aff_func(g(x)).

        Assumptions:
            The current tree and aff_func need to be compatible. Specifically, it is required that the output dimension
            of the current tree is equal to the input dimension of aff_func
        Notes:
            This function changes the output dimension of the current tree. The new output dimension is the output dimension
            of aff_func

            This function works only for affine functions. It can be considered a special case of the more general "compose"
            function which works for other AffTrees as well.
        """

    def evaluate(input: npt.NDArray) -> npt.NDArray:
        """
        Inputs
            input : npt.NDArray (Vector), an input for which g(input) is to be computed
        Outputs:
            output: npt.NDArray (Vector), the result g(input)

        Evaluates the currently represented function g on the input vector input.

        Assumptions:
            The dimension of input must match the input dimension of the current tree.
        """
        
    def compose(other: 'AffTree', prune: Optional[bool]):
        """
        Inputs
            other : AffTree, a tree representing a function f to be composed with the function g represented by the current tree
            prune (default: True) : Enable on-the-fly optimization during composition (recommended), see also infeasible_elimination

        Composes the current tree with other, specifically: After execution the current tree represents the function f(g(x)) where g
        is the function represented by the current tree and f is the function represented by other.


        Assumptions:
            Dimensions must match: Specifially, the output dimension of self must be equal to the input dimension of other.
        Notes:
            After composition, the current tree has outdimension equal to the outdimension of other.

            In the worst case, the size of the resulting tree can be proportional to the product of the original tree sizes.
            When composing two large trees, this can lead to long runtimes and large models. 
            To mitigate this one can either optimize the trees beforehand by calling infeasible_elimination or one can enable on-the-fly optimization for compose.
        """

    def infeasible_elimination():
        """
        Removes all infeasible paths from the tree. A path is called infeasible iff there exists no input that would ever
        traverse that path (for example, the path [x_1 <0, x_1 >0] is always infeasible). After execution, every edge in the
        tree can at least in principle be traversed by some input. In many cases, this drastically reduces the number of
        nodes and edges in the tree, saving resources.

        Optionally, a precondition can be supplied. In this case, paths are considered infeasible iff there exists no input
        that *satisfies that precondition* and traverses that path. This allows for a more aggressive path elimination,
        but leads to unknown behavior for any input that does not satisfy the precondition.


        Notes:
            Infeasible elimination does NOT change the semantics of the tree,
            but can drastically reduce its size.

            If infeasible elimination is not used, AffTrees can blow up very quickly. Therefore: Use early and often.
        """
        
    def size() -> int:
        """
        Outputs:
           size : int, the number of nodes in the tree

        Returns the number of nodes in this tree.
        """
        
    def depth() -> int:
        """
        Outputs:
           depth : int, the depth of the tree

        Returns the depth of the tree, that is, the number of nodes contained in the longest path from root to a terminal node.
        """
        
    def num_terminals() -> int:
        """
        Outputs:
           size : int, the number of terminals in the tree

        Returns the number of terminals in this tree.
        """
    
    def indim() -> int:
        pass
    
    def parent(node: 'AffNode') -> 'AffNode':
        """
        Inputs:
           node : AffNode, a node in the tree

        Returns the parent node of node. Can be used to iteratively traverse to the root of the tree.

        Assumptions:
            node should not be the root node as it has no parent.
        """
        
    def child(node: 'AffNode', label: int) -> 'AffNode':
        """
        Inputs:
           node : AffNode, a node in the tree
           label: int, the label specifying which child to retrieve
        Returns the child node of node. The label indicates which child to retrieve. Label=1 corresponds to the child
        where the condition of node is satisfied and label=0 to the child where the condition is not satisfied.

        Assumptions:
            node should not be a terminal node as it has no children.
        """
        
    def nodes() -> List['AffNode'] :
        """
        Outputs:
            nodes : List[AffNode], a list of all nodes in self

        Returns a list of all nodes contained in self.
        """
        
    def terminals() -> List['AffNode'] :
        """
        Outputs:
            nodes : List[AffNode], a list of all terminal nodes in self

        Returns a list of all terminal nodes contained in self.
        """

    def dfs() -> List[Tuple[int, 'AffNode', int]] :
        """
        Outputs:
            nodes : List[Tuple[int, AffNode, int]], a list of all nodes in self with their associated depth (first argument) and the number of its siblings that have not yet been vistied (last argument)

        Returns a list of all terminal nodes contained in self, associated with an integer that denotes
        the depth in which nodes were encountered in a depth-first search (https://en.wikipedia.org/wiki/Depth-first_search).
        """

    def edges() -> List[Tuple['AffNode', int, 'AffNode']] :
        """
        Outputs:
            edges : List[Tuple[AffNode, int, AffNode]], a list of all edges in self

        Returns a list of all edges in self. Edges are not specifically objects but are represented as tuples (source, label, target).
        """

    def polyhedra() -> List[Tuple[int, 'AffNode', 'Polytope']] :
        """
        Outputs:
            edges : List[Tuple[int, 'AffNode', 'Polytope']], a list of all linear regions in self

        Traverses the tree in a depth-first manner and returns for each encountered node its depth and the polyhedron implied by the path yielding to the node.

        Returns a list of all linear regions in self.
        Linear regions (https://en.wikipedia.org/wiki/Piecewise_linear_function#Notation) are represented as tuples:
        (index, node, Polytope) where index is a numbering of the linear regions, node is a terminal node of self that
        represents some affine function f and the Polytope characterizes the set of points where self agrees with that
        affine function.

        Notes:
            If infeasible paths are not eliminated, this method might return empty linear regions. To ensure that each
            linear region is indeed an actual linear region, call infeasibility elimination or check otherwise that each
            polyhedron is non-empty.
        """
        
    def __getitem__(key: int) -> 'AffNode' : ...

    def __str__() -> str : ...

    def __repr__() -> str : ...

class AffNode:
    """
    A node in an AffTree.

    Nodes always contain an AffineFunction f.

    Terminal nodes represent that AffineFunction,
    non-terminal nodes represent the implied linear predicate f(x) >= 0.
    """
    
    val: AffFunc
    id: int
    
    def is_terminal() -> bool: ...

    def is_decision() -> bool: ...

    def __richcmp__(other: 'AffNode', op: CompareOp) -> bool: ...

    def __hash__() -> int: ...

    def __repr__() -> str: ...

    def __str__() -> str: ...

class Polytope:
    """
    A class representing an n-dimensional Polytope (or Polytope https://en.wikipedia.org/wiki/Polytope).

    Polytopes are represented via a linear inequation system: P={x | Wx <= b}.

     Constructors
     ------------
     from_mats(weights:npt.NDArray (Matrix), bias: npt.NDArray (Vector)) -> Polytope
        Constructs the Polytope {x|weights*x <= bias}

    hyperrectangle(dim:int, Intervals : List[Tuple[float,float]]):
        Creates a polyhedron representing a dim-dimensional hyperrectangle.
        Intervals[i] should contain the lower and upper bound for values in the i-th dimension
    """
    
    @staticmethod
    def from_mats(weights: npt.NDArray, bias: npt.NDArray) -> 'Polytope': 
        """
        Inputs:
        ----------
        weights : npt.NDArray (Matrix n x m), a weight matrix
        bias    : npt.NDArray (Vector n), a bias vector

        ----------
        Creates the Polytope P={x | weights*x <= bias}.
        Dimension of the Polytope is automatically inferred from the input arguments.
        """

    @staticmethod
    def hyperrectangle(dim: int, intervals: List[Tuple[float, float]]) -> 'Polytope':
        """
        Inputs:
        ----------
        dim       :  int, the number of dimensions of this hyperrectangle
        intervals :  List[Tuple[float, float]], a list containing the lower- and upper-bounds per dimension

        ----------
        Creates a Polytope representing a dim-dimensional hyperrectangle with bounds specified in intervals.

        Mathematically, this corresponds to the set of points satisfying
        lower <= x_i <= upper
        for each i if (lower,upper)=intervals[i]
        """

    mat: npt.NDArray
    bias: npt.NDArray
    
    def indim() -> int: ...

    def n_constraints() -> int: ...

    def row(row: int) -> 'Polytope': ...

    def row_iter() -> List['Polytope']: ...

    def normalize() -> 'Polytope': ...

    def distance(point: npt.NDArray) -> npt.NDArray: ...

    def contains(point: npt.NDArray) -> bool: ...

    def translate(point: npt.NDArray) -> 'Polytope': ...

    def intersection(other: 'Polytope') -> 'Polytope': ...

    def rotate(array: npt.NDArray) -> 'Polytope': ...

    def slice(reference_vec: npt.NDArray, reduce_dim: Optional[bool]) -> 'Polytope': ...

    def chebyshev_center() -> Tuple['Polytope', npt.NDArray]: ...
    
    def solve(cost: Optional[npt.NDArray]) -> npt.NDArray: ...

    def __and__(other: 'Polytope') -> 'Polytope' : ...

    def __repr__() -> str : ...

    def __str__() -> str : ...

    def to_Axbleqz() -> Tuple[npt.NDArray, npt.NDArray]: ...

    def to_Axleqb() -> Tuple[npt.NDArray, npt.NDArray]: ...

    def to_Axbgeqz() -> Tuple[npt.NDArray, npt.NDArray]: ...

    def to_Axgeqb() -> Tuple[npt.NDArray, npt.NDArray]: ...

class AffFunc:
    """
    A class used to represent affine (or linear) functions.

    Linear functions are of the form f: R^n -> R^m with f(x)=W*x+b where b is some bias vector R^m and W is a matrix R^{m x n}.

    An affine function is therefore fully characterized by W and b.
    """
    
    @staticmethod
    def from_mats(weights: npt.NDArray, bias: npt.NDArray) -> 'AffFunc': ...

    @staticmethod
    def identity(dim: int) -> 'AffFunc': ...
    
    @staticmethod
    def zeros(dim: int) -> 'AffFunc': ...

    @staticmethod
    def constant(dim: int, value: float) -> 'AffFunc': ...

    @staticmethod
    def unit(dim: int, column: int) -> 'AffFunc': ...

    @staticmethod
    def unit(dim: int, column: int) -> 'AffFunc': ...

    @staticmethod
    def zero_idx(dim: int, index: int) -> 'AffFunc': ...

    @staticmethod
    def rotation(orthogonal_mat: npt.NDArray) -> 'AffFunc': ...

    @staticmethod
    def uniform_scaling(dim: int, scalar: float) -> 'AffFunc': ...

    @staticmethod
    def scaling(scalars: npt.NDArray) -> 'AffFunc': ...

    @staticmethod
    def slice(reference_point: npt.NDArray) -> 'AffFunc': ...

    @staticmethod
    def translation(dim: int, offset: npt.NDArray) -> 'AffFunc': ...

    mat: npt.NDArray
    bias: npt.NDArray

    def indim() -> int: ...

    def outdim() -> int: ...

    def row(row: int) -> 'AffFunc': ...

    def row_iter() -> 'AffFunc': ...

    def apply(input: npt.NDArray) -> npt.NDArray: ...

    def apply_transpose(input: npt.NDArray) -> npt.NDArray: ...

    def compose(other: 'AffFunc') -> 'AffFunc': ...

    def stack(other: 'AffFunc') -> 'AffFunc': ...

    def add(other: 'AffFunc') -> 'AffFunc': ...

    def negate() -> 'AffFunc': ...

    def __add__(other: 'AffFunc') -> 'AffFunc': ...

    def __neg__() -> 'AffFunc': ...
