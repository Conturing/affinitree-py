#   Copyright 2024 affinitree developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

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
        Constructs a simple AffTree representing the identity function in dim dimensions.
        Evaluating this tree yields f(x)=x.
        
        Parameters
        ----------
        dim : int
            The input dimension of this AffTree.
        
        Returns
        -------
        AffTree
            An instance of AffTree that represents the identity function of dim `dim`.
        """
        
    @classmethod
    def from_aff(cls, func: AffFunc) -> 'AffTree':
        """
        Constructs an AffTree from a given affine function represented by `func`.
        
        Parameters
        ----------
        func : AffFunc
            An affine function object that defines both the linear transformation (matrix)
            and translation (vector) components to be represented by the tree.
        
        Returns
        -------
        AffTree
            An instance of AffTree that represents the affine function described by `func`.
        """
    
    @classmethod
    def from_array(cls, weights: npt.NDArray, bias: npt.NDArray) -> 'AffTree':
        """
        Constructs a simple AffTree representing the function f(x)=weights*x + bias.
        
        Parameters
        ----------
        weights : npt.NDArray (Matrix n x m)
            Weight matrix
        bias : npt.NDArray (Vector n), 
            Bias vector

        Assumptions
        -----------
        Ensure that the dimensions of the weights and bias are compatible, i.e., weights.shape[0]=bias.shape[0]
        """
    
    @classmethod
    def from_poly(cls, precondition: Polytope, func_true: AffFunc, func_false: Optional[AffFunc]) -> AffTree:
         """
        Constructs an AffTree based on a polytope precondition and associated affine functions.

        This method creates an AffTree where the decisions are based on the rows (hyperplanes)
        of the given polytope. Depending on whether the input satisfies the precondition, 
        the tree evaluates to `func_true` or `func_false`. If `func_false` is not provided, 
        the tree is undefined for inputs not satisfying the precondition.

        Parameters
        ----------
        precondition : Polytope
            A polytope that defines the precondition for splitting the input space.
        func_true : AffFunc
            The affine function to be used when the precondition is satisfied.
        func_false : AffFunc, optional
            The affine function to be used when the precondition is not satisfied (default is None,
            which results in a partial function).

        Returns
        -------
        AffTree
            An instance of AffTree that represents the piece-wise affine function defined by
            the polytope condition and the given affine functions.
        """
        
    def apply_func(func: 'AffFunc'):
        """
        Applies an affine function to the AffTree.

        This method modifies the current AffTree by applying the given affine function `func`.
        It essentially composes `func` with the existing tree's function. The result is a new
        AffTree that, when evaluated, will compute the result of the `func` applied to the output
        of the original tree function.

        Parameters
        ----------
        func : AffFunc
            An affine function that will be applied to the outputs of the current tree. This function
            must have an input dimension that matches the output dimension of the tree.
        """

    def evaluate(input: npt.NDArray) -> npt.NDArray:
        """
        Evaluates the currently represented function g of this tree on the given input vector input.
        
        Parameters
        ----------
        input : npt.NDArray (Vector)
            An input for which g(input) is to be computed.
        
        Returns
        -------
        npt.NDArray (Vector)
            The result g(input).

        Assumptions
        -----------
        The dimension of input must match the input dimension of the current tree.
        """
        
    def compose(other: 'AffTree', prune: Optional[bool]):
        """
        Composes the current tree with other.
        
        After execution the current tree represents the function f(g(x)) where g
        is the function represented by the current tree and f is the function represented by other.
        This is the opposite order of (mathematical) function composition as, e.g., used by AffFunc.compose.
        
        Parameters
        ----------
        other : AffTree
            A tree representing a function f to be composed with the function g represented by the current tree.
        prune : bool (default: True) 
            Enable on-the-fly optimization during composition (recommended), see also infeasible_elimination.

        Assumptions
        -----------
        Dimensions must match: Specifially, the output dimension of self must be equal to the input dimension of other.
        
        Notes
        -----
        After composition, the current tree has out dimension equal to the out dimension of other.

        In the worst case, the size of the resulting tree can be proportional to the product of the original tree sizes.
        When composing two large trees, this can lead to long runtimes and large models. 
        To mitigate this one can either optimize the trees beforehand by calling infeasible_elimination or one can enable on-the-fly optimization for compose.
        """

    def infeasible_elimination():
        """
        Removes all infeasible paths from the tree. 
        
        A path is called infeasible iff there exists no input that would ever
        traverse that path (for example, the path [x_1 <0, x_1 >0] is always infeasible). After execution, every edge in the
        tree can at least in principle be traversed by some input. In many cases, this drastically reduces the number of
        nodes and edges in the tree, saving resources.

        Notes
        -----
        Infeasible elimination does NOT change the semantics of the tree,
        but can drastically reduce its size.

        If infeasible elimination is not used, AffTrees can blow up very quickly. Therefore: Use early and often.
        """
        
    def size() -> int:
        """
        Returns the number of nodes in this tree.
        
        Returns
        -------
        size : int
            The number of nodes in the tree.
        """
        
    def depth() -> int:
        """
        Returns the depth of the tree, that is, the number of nodes contained in the longest path from root to a terminal node.
        
        Returns
        -------
        depth : int
            The depth of the tree.
        """
        
    def num_terminals() -> int:
        """
        Returns the number of terminals in this tree.
        
        Returns
        -------
        size : int
            The number of terminals in the tree.
        """
    
    def indim() -> int:
        """
        Returns the input dimension of the AffTree.
        """
    
    def parent(node: 'AffNode') -> 'AffNode':
        """
        Returns the parent node of node. Can be used to iteratively traverse to the root of the tree.
        
        Parameters
        ----------
        node : AffNode
            A node of this tree.

        Assumptions
        -----------
        Node should not be the root node as it has no parent.
        """
        
    def child(node: 'AffNode', label: int) -> 'AffNode':
        """
        Returns the child node of node. The label indicates which child to retrieve. Label=1 corresponds to the child
        where the condition of node is satisfied and label=0 to the child where the condition is not satisfied.
        
        Parameters
        ----------
        node : AffNode
            A node in the tree.
        label: int
            The label specifying which child to retrieve.
        
        Assumptions
        -----------
        Node should not be a terminal node as it has no children.
        """
        
    def nodes() -> List['AffNode']:
        """
        Returns a list of all nodes contained in self in memory order.
        
        Returns
        -------
        nodes : list of AffNode
            A list of all nodes in this tree.
        """
        
    def terminals() -> List['AffNode']:
        """
        Returns a list of all terminal nodes contained in self in memory order.
        
        Returns
        -------
        nodes : list of AffNode
            A list of all terminal nodes in this tree.
      
        """

    def dfs() -> List[Tuple[int, 'AffNode', int]]:
        """
        Returns a list of all terminal nodes contained in self, associated with an integer that denotes
        the depth in which nodes were encountered in a depth-first search (https://en.wikipedia.org/wiki/Depth-first_search).
        
        Returns
        -------
        nodes : List[Tuple[int, AffNode, int]], 
            A list of all nodes in this tree with their associated depth (first argument) and the number of its siblings that have not yet been vistied (last argument).
        """

    def edges() -> List[Tuple['AffNode', int, 'AffNode']]:
        """
        Returns a list of all edges in self. Edges are not specifically objects but are represented as tuples (source, label, target).
        
        Returns
        -------
        edges : List[Tuple[AffNode, int, AffNode]]
            A list of all edges in this tree.
        """

    def polyhedra() -> List[Tuple[int, 'AffNode', 'Polytope']]:
        """
        Traverses the tree in a depth-first manner and returns for each encountered node its depth and the polytope of the path yielding to the node.

        Returns a list of all linear regions in self.
        Linear regions (https://en.wikipedia.org/wiki/Piecewise_linear_function#Notation) are represented as tuples:
        (depth, node, poly) where depth is the depth in the tree, node is a node in the tree, and poly is a Polytope characterizes the region that is associated with this node (i.e., all inputs that take the path to the node).

        Returns
        -------
        edges : List[Tuple[int, 'AffNode', 'Polytope']]
            A list of all linear regions in this tree.

        Notes
        -----
        If infeasible paths are not eliminated, this method might return empty linear regions. To ensure that each
        linear region is indeed an actual linear region, call infeasibility elimination or check otherwise that each
        polyhedron is non-empty.
        """
    
    def reduce():
        """
        Reduces the complexity of the AffTree by merging equivalent terminal nodes.

        This method optimizes the structure of the AffTree by identifying and merging terminals that represent
        equivalent affine functions.
        The reduction process aims to minimize the size of the AffTree without changing its functional behavior.
        """
    
    def remove_axes(mask: npt.ArrayLike):
        """
        Removes specified axes from the AffTree based on the provided mask.

        This method modifies the AffTree by eliminating dimensions (or axes) of the input space that are specified by `mask`. The `mask` should be an array-like structure where each element
        corresponds to a dimension in the input space of the AffTree, with `False` values indicating dimensions to
        remove.

        Parameters
        ----------
        mask : npt.ArrayLike
            An array-like structure indicating which axes of the input space should be removed.
        """
    
    def to_dot() -> str:
        """
        Converts the AffTree to a DOT format string.

        This method generates a string representation of the AffTree in DOT format, which can be used for
        visualization with Graphviz.

        Returns
        -------
        str
            A string that represents the entire AffTree in DOT format.
        """
    
    def __add__(other: 'AffTree') -> 'AffTree' : ...
    
    def __sub__(other: 'AffTree') -> 'AffTree' : ...
    
    def __mul__(other: 'AffTree') -> 'AffTree' : ...
    
    def __div__(other: 'AffTree') -> 'AffTree' : ...

    def __neg__() -> 'AffTree' : ...
    
    def __getitem__(key: int) -> 'AffNode' : ...

    def __str__() -> str : ...

    def __repr__() -> str : ...

class AffNode:
    """
    A node in an AffTree.

    Nodes always contain an AffineFunction f.

    Terminal nodes represent that AffineFunction,
    non-terminal nodes represent the implied linear predicate f(x) >= 0.
    
    Attributes
    ----------
    val: AffFunc
        Function object representing the linear property of this node
        (either linear function or linear decision)
    id: int
        Unique id of this node used for referencing.
    """
    
    val: AffFunc
    id: int
    
    def is_terminal() -> bool:
        """
        Returns whether or not this node is a terminal node in its AffTree.
        """

    def is_decision() -> bool:
         """
        Returns whether or not this node is a inner node in its AffTree.
        """

    def __richcmp__(other: 'AffNode', op: CompareOp) -> bool: ...

    def __hash__() -> int: ...

    def __repr__() -> str: ...

    def __str__() -> str: ...

class Polytope:
    """
    Represents an n-dimensional polytope.
    
    A polytope is a geometric object with flat sides, which generalizes the notion of polyhedra to n dimensions.
    A polytope is defined as the solution set to a system of linear inequalities, represented by:
    P = {x | mat @ x <= bias}, where 'mat' is a coefficient matrix and 'bias' is a vector of constants.

    For more details, see https://en.wikipedia.org/wiki/Polytope.

    Constructors
    ------------
    from_mats(weights:npt.NDArray (Matrix), bias: npt.NDArray (Vector)) -> Polytope
        Constructs the Polytope {x | mat @ x <= bias}

    hyperrectangle(dim:int, Intervals : List[Tuple[float,float]]):
        Creates a polyhedron representing a dim-dimensional hyperrectangle.
        Intervals[i] should contain the lower and upper bound for values in the i-th dimension
    
    Attributes
    ----------
    mat: npt.NDArray
        The matrix of coefficients for the linear inequalities defining the polytope.
    bias: npt.NDArray
        The vector of constants for the linear inequalities defining the polytope.
    """
    
    @staticmethod
    def from_mats(mat: npt.NDArray, bias: npt.NDArray) -> 'Polytope': 
        """
        Creates the Polytope from matrices.
        
        This method creates a Polytope defined by a system of linear inequalities of the form:
        P = {x | mat @ x <= bias}.
        Dimension of the Polytope is automatically inferred from the input arguments.
        
        Parameters
        ----------
        mat : npt.NDArray (Matrix n x m)
            A matrix of coefficients.
        bias : npt.NDArray (Vector n)
            A vector of constants.
        """

    @staticmethod
    def hyperrectangle(dim: int, intervals: List[Tuple[float, float]]) -> 'Polytope':
        """
        Creates a Polytope representing a hyperrectangle, defined by dimension-wise intervals.

        This method creates a hyperrectangle in an n-dimensional space specified by intervals for each dimension. Each
        interval defines the lower and upper bounds for that dimension, constructing a Polytope that encompasses
        all points within these bounds.

        Parameters
        ----------
        dim : int
            The dimensionality of the hyperrectangle.
        intervals : List[Tuple[float, float]]
            A list containing the lower- and upper-bounds per dimension.

        Returns
        -------
        polytope
            A new instance of Polytope that represents the hyperrectangle defined by the given intervals.

        """
    
    mat: npt.NDArray
    bias: npt.NDArray
    
    def indim() -> int:
        """
        Returns the input dimension of the Polytope.
        """

    def n_constraints() -> int:
        """
        Returns the number of linear constraints (inequalities) defining the Polytope.
        
        Each row in `self.mat` represents a single linear constraint that contributes to defining
    the boundaries of the Polytope.
        """

    def row(row: int) -> 'Polytope':
        """
        Constructs a new one-dimensional Polytope from a specific constraint row of the inequality system.

        Parameters
        ----------
        row : int
            The index of the row in the inequality system defining this Polytope.

        Returns
        -------
        Polytope
            A new instance of Polytope representing the constraint defined by the selected row.

        Raises
        ------
        IndexError
            If the specified row index is out of the bounds.
        """

    def row_iter() -> List['Polytope']: ...

    def normalize() -> 'Polytope':
        """
        Normalizes the coefficients and constants of the Polytope to have unit norms.

        This method adjusts the rows of the coefficients matrix and the corresponding constants so that
        each constraint (row) in the Polytope has a unit norm. This normalization often simplifies
        numerical calculations and comparisons.

        Returns
        -------
        Polytope
            A new instance of Polytope with normalized constraints.
        """

    def distance(point: npt.NDArray) -> npt.NDArray:
        """
        Calculates the distance from a given point to each of the halfspaces defined by the Polytope.

        This method computes the signed distances from the specified point to the halfspaces defined by each row 
        in the coefficients matrix combined with the constants vector. These halfspaces represent the linear inequalities
        that define the boundaries of the Polytope. The distance indicates how far the point is from satisfying each
        inequality, with negative values indicating the point is outside the corresponding halfspace.

        Parameters
        ----------
        point : npt.NDArray
            An array representing a point in the space where the Polytope is defined.

        Returns
        -------
        npt.NDArray
            An array of distances, where each element corresponds to the distance from the point to a specific halfspace
            defined by the Polytope.
        """

    def contains(point: npt.NDArray) -> bool:
         """
        Determines if a given point is inside the Polytope.

        This method evaluates whether the specified point satisfies all the linear inequalities 
        defined by the Polytope. If the point meets all conditions (mat * `point` <= bias), 
        it is considered to be inside the Polytope.

        Parameters
        ----------
        point : npt.NDArray
            An array representing a point in the space where the Polytope is defined.

        Returns
        -------
        bool
            True if the point is inside the Polytope, False otherwise.
        """

    def translate(point: npt.NDArray) -> 'Polytope': ...

    def intersection(other: 'Polytope') -> 'Polytope': ...

    def rotate(array: npt.NDArray) -> 'Polytope': ...

    def slice(reference_vec: npt.NDArray, reduce_dim: Optional[bool]) -> 'Polytope': ...

    def chebyshev_center() -> Tuple['Polytope', npt.NDArray]: ...
    
    def solve(cost: Optional[npt.NDArray]) -> npt.NDArray:
        """
        Solves an optimization problem over the Polytope, minimizing a given cost function (linear program).

        This method finds the point within the Polytope that minimizes the cost function specified by the 'cost' array.
        If no cost array is provided, the method defaults to a feasibility check with costs equal to zero.

        Parameters
        ----------
        cost : npt.NDArray, optional
            An array representing the cost coefficients for each dimension of the Polytope.

        Returns
        -------
        npt.NDArray
            An array representing the point within the Polytope that optimizes the specified cost function.
        
        Raises
        ------
        ValueError
            If no optimal point can be found, e.g., because the Poyltope is empty or unbounded. 
        """

    def __and__(other: 'Polytope') -> 'Polytope' : ...

    def __repr__() -> str : ...

    def __str__() -> str : ...

    def to_Axbleqz() -> Tuple[npt.NDArray, npt.NDArray]: ...

    def to_Axleqb() -> Tuple[npt.NDArray, npt.NDArray]: ...

    def to_Axbgeqz() -> Tuple[npt.NDArray, npt.NDArray]: ...

    def to_Axgeqb() -> Tuple[npt.NDArray, npt.NDArray]: ...

class AffFunc:
    """
    Represents an affine (linear) function.

    Linear functions are of the form f: R^n -> R^m with f(x) = mat @ x + bias where
    mat is a matrix of coefficients and bias is a vector of constants.
    
    Attributes
    ----------
    mat: npt.NDArray
        The matrix of coefficients of this function.
    bias: npt.NDArray
        The vector of constants of this function.
    """
    
    @staticmethod
    def from_mats(mat: npt.NDArray, bias: npt.NDArray) -> 'AffFunc':
        """
        Constructs an AffFunc instance from a matrix of coefficients and a vector of constants.

        This method initializes an affine function f: R^n -> R^m, where the function is defined as f(x) = mat @ x + bias.
        The 'mat' provides the coefficients for the linear transformation, and 'bias' adds a constant vector to the results
        of this transformation, defining the affine function fully.

        Parameters
        ----------
        mat : npt.NDArray
            A matrix representing the coefficients of the linear part of the affine function.
        bias : npt.NDArray
            A vector representing the constants added to the linear transformation.
            It should have a dimension matching the number of rows in 'mat'.

        Returns
        -------
        AffFunc
            An instance of AffFunc representing the specified affine function.
        """

    @staticmethod
    def identity(dim: int) -> 'AffFunc':
        """
        Constructs an identity affine function of specified dimension.

        This method creates an affine function that acts as the identity function over R^dim, meaning it returns
        the input as the output without any transformation.

        Parameters
        ----------
        dim : int
            The dimensionality of the input and output space of the affine function.

        Returns
        -------
        AffFunc
            An instance of AffFunc representing the identity function for the specified dimension.
        """
    
    @staticmethod
    def zeros(dim: int) -> 'AffFunc': ...

    @staticmethod
    def constant(dim: int, value: float) -> 'AffFunc': ...

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

    def apply(input: npt.NDArray) -> npt.NDArray:
        """
        Applies this affine function to a given input vector.

        Computes the value of this function for the given point `input`, i.e.,
        calculates mat @ input + bias.

        Parameters
        ----------
        input : npt.NDArray
            An input vector to be transformed. 

        Returns
        -------
        npt.NDArray
            The output vector after applying the affine transformation to the input.
        """

    def apply_transpose(input: npt.NDArray) -> npt.NDArray: ...

    def compose(other: 'AffFunc') -> 'AffFunc':
        """
        Composes this affine function with another, producing a new affine function via sequential evaluation.

        This method achieves function composition, denoted mathematically as f \circ g, where `f` is this function and `g` is the other function. 
        The resulting composition is h(x) = f(g(x)).

        Parameters
        ----------
        other : AffFunc
            The other affine function to compose with this function. It is crucial that the output dimension of `other` matches the input dimension
            of this function to ensure the composition is well-defined.

        Returns
        -------
        AffFunc
            A new AffFunc instance representing the composition of this function followed by the other function.
        """

    def stack(other: 'AffFunc') -> 'AffFunc': ...

    def __add__(other: 'AffFunc') -> 'AffFunc': ...
    
    def __sub__(other: 'AffFunc') -> 'AffFunc': ...
    
    def __mul__(other: 'AffFunc') -> 'AffFunc': ...
    
    def __div__(other: 'AffFunc') -> 'AffFunc': ...
    
    def __mod__(other: 'AffFunc') -> 'AffFunc': ...

    def __neg__() -> 'AffFunc': ...


class LayerBuilder:
    """
    A builder for creating a neural network layer-by-layer with specific activation functions and transformations.

    Attributes
    ----------
    layers : List[Any]
        The list of layers in the builder.
    input_dim : int
        The dimension of the input to the first layer.
    current_dim : int
        The current output dimension after the last added layer.
    """

    def __init__(self, input_dim: int) -> None:
        """
        Initializes the LayerBuilder with the specified input dimension.

        Parameters
        ----------
        input_dim : int
            The dimension of the input data expected by the layers.
        """

    def linear(self, aff: Any) -> None:
        """
        Adds a linear layer to the network.

        Parameters
        ----------
        aff : Any
            An affine function wrapper that contains the necessary linear transformation.
        """

    def partial_relu(self, idx: int) -> None:
        """
        Adds a ReLU activation to a specific neuron.

        Parameters
        ----------
        idx : int
            Index of the neuron to apply ReLU.
        """

    def relu(self) -> None:
        """
        Adds a ReLU activation to all neurons in the current layer.
        """

    def partial_leaky_relu(self, idx: int, alpha: float) -> None:
        """
        Adds a leaky ReLU activation to a specific neuron with a given slope coefficient.

        Parameters
        ----------
        idx : int
            Index of the neuron to apply leaky ReLU.
        alpha : float
            Slope coefficient for the leaky part of ReLU.
        """

    def leaky_relu(self, alpha: float) -> None:
        """
        Adds a leaky ReLU activation to all neurons in the current layer with a given slope coefficient.

        Parameters
        ----------
        alpha : float
            Slope coefficient for the leaky part of ReLU.
        """

    def partial_hard_tanh(self, idx: int) -> None:
        """
        Adds a hard tangent hyperbolic activation to a specific neuron.

        Parameters
        ----------
        idx : int
            Index of the neuron to apply hard tanh.
        """

    def hard_tanh(self) -> None:
        """
        Adds a hard tangent hyperbolic activation to all neurons in the current layer.
        """

    def partial_hard_sigmoid(self, idx: int) -> None:
        """
        Adds a hard sigmoid activation to a specific neuron.

        Parameters
        ----------
        idx : int
            Index of the neuron to apply hard sigmoid.
        """

    def hard_sigmoid(self) -> None:
        """
        Adds a hard sigmoid activation to all neurons in the current layer.
        """

    def argmax(self) -> None:
        """
        Applies an argmax operation, effectively reducing the dimension to 1 by selecting the maximum index.
        """

    def __str__(self) -> str:
        """
        Returns a string representation of the current layers in the builder.

        Returns
        -------
        str
            A formatted string listing all the layers with details.
        """