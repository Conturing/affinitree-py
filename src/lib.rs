//   Copyright 2024 affinitree developers
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#![allow(unused)]

use pyo3::exceptions::{PyValueError, PyNotImplementedError};
use pyo3::prelude::*;
use pyo3::panic::PanicException;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyType, PySlice};

use numpy::{IntoPyArray, Ix1, PyArray, PyArray1, PyArrayDyn, PyReadonlyArray, PyReadonlyArray1, PyReadonlyArray2, PyArray2, PyUntypedArray};
use numpy::ndarray::{Array1, Array2, Axis};

use pyo3_arraylike::{PyArrayLike1, PyArrayLike2};

use affinitree_rust::linalg::polyhedron::PolytopeStatus;
use affinitree_rust::linalg::affine::{AffFunc, Polytope};
use affinitree_rust::tree::graph::{Tree, TreeNode, TreeIndex, Label};
use affinitree_rust::pwl::afftree::AffTree;
use affinitree_rust::pwl::node::{AffContent, AffNode};
use affinitree_rust::pwl::dot::{self, Dot};
use affinitree_rust::distill::builder::{afftree_from_layers, read_layers, self, Layer, afftree_from_layers_verbose, afftree_from_layers_csv};
use affinitree_rust::distill::schema;


#[derive(FromPyObject)]
pub enum SliceOrInt<'a> {
    Slice(&'a PySlice),
    Int(isize),
}

fn afffunc<'py>(mat: PyArrayLike2<'py, f64>, bias: PyArrayLike1<'py, f64>) -> Result<AffFunc, PyErr> {
    let mat = mat.into_owned_array();
    let bias = bias.into_owned_array();
    if mat.shape()[0] != bias.shape()[0] {
        Err(PyValueError::new_err(format!("Dimensions mismatch: {} vs {}", mat.shape()[0], bias.shape()[0])))
    } else {
        Ok(AffFunc::from_mats(mat, bias))
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "AffTree")]
struct AffTreeWrapper {
    aff_tree: AffTree<2>,
}

#[pymethods]
impl AffTreeWrapper {

    #[classmethod]
    fn identity(cls: &PyType, dim: usize) -> PyResult<AffTreeWrapper> {
        Ok(AffTreeWrapper { aff_tree: AffTree::from_aff(AffFunc::identity(dim)) })
    }

    #[classmethod]
    fn from_aff(cls: &PyType, func: &AffineFunctionWrapper) -> PyResult<AffTreeWrapper> {
        Ok(AffTreeWrapper { aff_tree: AffTree::from_aff(func.aff_func.clone()) })
    }

    #[classmethod]
    fn from_array<'py>(cls: &PyType, weights: PyArrayLike2<'py, f64>, bias: PyArrayLike1<'py, f64>) -> PyResult<AffTreeWrapper> {
        Ok(AffTreeWrapper { aff_tree: AffTree::from_aff(afffunc(weights, bias)?) })
    }

    #[classmethod]
    fn from_poly<'py>(cls: &PyType, precondition: PolytopeWrapper, func_true: AffineFunctionWrapper, func_false: Option<AffineFunctionWrapper>) -> PyResult<AffTreeWrapper> {
        Ok(AffTreeWrapper { aff_tree: AffTree::from_poly(precondition.polytope, func_true.aff_func, func_false.map(|w| w.aff_func).as_ref()) })
    }

    fn apply_func(&mut self, aff_func: &AffineFunctionWrapper) {
        self.aff_tree.apply_func(&aff_func.aff_func);
    }

    fn evaluate<'py>(&self, _py: Python<'py>, input: PyArrayLike1<'py, f64>) -> PyResult<&'py PyArray1<f64>> {
        let res = self.aff_tree.evaluate(&input.into_owned_array());
        match res {
            Some(val) => Ok(PyArray::from_array(_py, &val)),
            None => Err(PyValueError::new_err("Evaluation failed: node as no child for given input"))
        }
    }

    fn compose(&mut self, other: &AffTreeWrapper, prune: Option<bool>) {
        if let Some(false) = prune {
            self.aff_tree.compose::<false>(&other.aff_tree);
        } else {
            self.aff_tree.compose::<true>(&other.aff_tree);
        }
    }

    fn infeasible_elimination(&mut self) {
        self.aff_tree.infeasible_elimination();
    }

    #[getter]
    fn root(&self) -> PyResult<AffineNodeWrapper> {
        Ok(AffineNodeWrapper::new(self.aff_tree.tree.get_root_idx(), self.aff_tree.tree.get_root().clone()))
    }

    fn size(&self) -> PyResult<usize> {
        Ok(self.aff_tree.len())
    }

    fn depth(&self) -> PyResult<usize> {
        Ok(self.aff_tree.tree.depth())
    }

    fn num_terminals(&self) -> PyResult<usize> {
        Ok(self.aff_tree.num_terminals())
    }

    fn indim(&self) -> PyResult<usize> {
        Ok(self.aff_tree.in_dim)
    }

    fn parent(&self, node: &AffineNodeWrapper) -> PyResult<AffineNodeWrapper> {
        let res = self.aff_tree.tree.parent(node.idx);
        match res {
            Some(edg) => Ok(AffineNodeWrapper::new(edg.source_idx, self.aff_tree.tree.tree_node(edg.source_idx).unwrap().clone())),
            None => Err(PyValueError::new_err(format!("No node with id {} exists in graph", node.idx)))
        }
    }

    fn child(&self, node: &AffineNodeWrapper, label: usize) -> PyResult<AffineNodeWrapper> {
        let res = self.aff_tree.tree.child(node.idx, label);
        match res {
            Some(edg) => Ok(AffineNodeWrapper::new(edg.target_idx, self.aff_tree.tree.tree_node(edg.target_idx).unwrap().clone())),
            None => Err(PyValueError::new_err(format!("No node with id {} exists in graph", node.idx)))
        }
    }

    fn nodes(&self) -> PyResult<Vec<AffineNodeWrapper>> {
        Ok(self.aff_tree.tree.node_iter().map(|(idx, node)| AffineNodeWrapper {node: node.clone(), idx}).collect())
    }

    fn terminals(&self) -> PyResult<Vec<AffineNodeWrapper>> {
        Ok(self.aff_tree.tree.terminals().map(|nd| {
            let node = self.aff_tree.tree.tree_node(nd.idx).unwrap();
            AffineNodeWrapper::new(nd.idx, node.clone())
        }).collect())
    }

    fn dfs(&self) -> PyResult<Vec<(usize, AffineNodeWrapper, usize)>> {
        Ok(self.aff_tree.tree.dfs_iter()
            .map(|data| (data.depth, AffineNodeWrapper::new(data.index, self.aff_tree.tree.tree_node(data.index).unwrap().clone()), data.n_remaining)).collect())
    }

    fn edges(&self) -> PyResult<Vec<(AffineNodeWrapper, Label, AffineNodeWrapper)>> {
        Ok(self.aff_tree.tree.edge_iter().map(|edg| {
            let src_node = self.aff_tree.tree.tree_node(edg.source_idx).unwrap();
            let tgt_node = self.aff_tree.tree.tree_node(edg.target_idx).unwrap();
            (AffineNodeWrapper::new(edg.source_idx, src_node.clone()), edg.label, AffineNodeWrapper::new(edg.target_idx, tgt_node.clone()))
        }).collect())
    }

    fn polyhedra(&self) -> PyResult<Vec<(usize, AffineNodeWrapper, PolytopeWrapper)>> {
        Ok(self.aff_tree.polyhedra_iter()
            .map(|(depth, idx, n_remaining, poly)| 
                (depth, 
                AffineNodeWrapper::new(idx, self.aff_tree.tree.tree_node(idx).unwrap().clone()), 
                PolytopeWrapper { polytope: Polytope::intersection_n(self.aff_tree.in_dim(), poly.as_slice())}
                ))
            .collect())
    }

    fn reduce(&mut self) {
        self.aff_tree.reduce()
    }

    fn remove_axes<'py>(&mut self, _py: Python<'py>, mask: PyArrayLike1<'py, bool>) {
        self.aff_tree.remove_axes(&mask.into_owned_array())
    }

    fn to_dot(&self) -> String {
        let dot = Dot::from(&self.aff_tree);
        dot.to_string()
    }

    pub fn __neg__(&self) -> AffTreeWrapper {
        AffTreeWrapper { aff_tree: -self.aff_tree.clone() }
    }

    pub fn __add__(&self, other: &AffTreeWrapper) -> AffTreeWrapper {
        AffTreeWrapper { aff_tree: &self.aff_tree - &other.aff_tree }
    }

    pub fn __sub__(&self, other: &AffTreeWrapper) -> AffTreeWrapper {
        AffTreeWrapper { aff_tree: &self.aff_tree - &other.aff_tree }
    }

    pub fn __mul__(&self, other: &AffTreeWrapper) -> AffTreeWrapper {
        AffTreeWrapper { aff_tree: &self.aff_tree * &other.aff_tree }
    }

    pub fn __div__(&self, other: &AffTreeWrapper) -> AffTreeWrapper {
        AffTreeWrapper { aff_tree: &self.aff_tree / &other.aff_tree }
    }

    fn __getitem__(&self, key: TreeIndex) -> PyResult<AffineNodeWrapper> {
        match self.aff_tree.tree.tree_node(key) {
            Some(x) => Ok(AffineNodeWrapper::new(key, x.clone())),
            None => Err(PyValueError::new_err(format!("No node at index {}", key)))
        }
    }

    fn __str__(&self) -> String {
        self.aff_tree.to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.aff_tree)
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "AffNode")]
struct AffineNodeWrapper {
    node: AffNode<2>,
    idx: TreeIndex
}

impl AffineNodeWrapper {
    fn new(idx: TreeIndex, node: AffNode<2>) -> AffineNodeWrapper {
        AffineNodeWrapper { node, idx }
    }
}

#[pymethods]
impl AffineNodeWrapper {

    #[getter]
    fn val(&self) -> PyResult<AffineFunctionWrapper> {
        Ok(AffineFunctionWrapper { aff_func: self.node.value.aff.clone() })
    }

    #[getter]
    fn id(&self) -> PyResult<usize> {
        Ok(self.idx)
    }

    fn is_terminal(&self) -> PyResult<bool> {
        Ok(self.node.isleaf)
    }

    fn is_decision(&self) -> PyResult<bool> {
        Ok(!self.node.isleaf)
    }

    fn __richcmp__(&self, other: PyRef<AffineNodeWrapper>, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.idx == other.idx),
            CompareOp::Ne => Ok(self.idx != other.idx),
            _ => Err(PyNotImplementedError::new_err(""))
        }
    }

    fn __hash__(&self) -> usize {
        self.idx
    }

    fn __repr__(&self) -> String {
        format!("({:>4}, {})", self.idx, self.node)
    }

    fn __str__(&self) -> String {
        self.node.to_string()
    }

}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Polytope")]
struct PolytopeWrapper {
    polytope: Polytope,
}

impl PolytopeWrapper {
    fn new(poly: Polytope) -> PolytopeWrapper {
        PolytopeWrapper {
            polytope: poly
        }
    }
}

#[pymethods]
impl PolytopeWrapper {
    
    #[staticmethod]
    fn from_mats<'py>(weights: PyArrayLike2<'py, f64>, bias: PyArrayLike1<'py, f64>) -> PyResult<PolytopeWrapper> {
        let polytope = Polytope::from_mats(weights.into_owned_array(), bias.into_owned_array());
        Ok(PolytopeWrapper::new(polytope))
    }

    #[staticmethod]
    fn hyperrectangle(dim: usize, intervals: Vec<(f64, f64)>) -> PyResult<PolytopeWrapper> {
        let polytope = Polytope::hyperrectangle(dim, intervals.as_slice());
        Ok(PolytopeWrapper::new(polytope))
    }

    fn indim(&self) -> PyResult<usize> {
        Ok(self.polytope.indim())
    }

    fn n_constraints(&self) -> PyResult<usize> {
        Ok(self.polytope.outdim())
    }

    #[getter]
    fn mat<'py>(&self, _py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        Ok(PyArray::from_array(_py, &self.polytope.matrix_view()))
    }

    #[getter]
    fn bias<'py>(&self, _py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
        Ok(PyArray::from_array(_py, &self.polytope.bias_view()))
    }

    fn row(&self, row: usize) -> PyResult<PolytopeWrapper> {
        Ok(PolytopeWrapper::new(self.polytope.row(row).to_owned()))
    }

    fn row_iter(&self) -> PyResult<Vec<PolytopeWrapper>> {
        Ok(self.polytope.row_iter().map(|x| PolytopeWrapper::new(x.to_owned())).collect())
    }

    fn normalize(&self) -> PyResult<PolytopeWrapper> {
        Ok(PolytopeWrapper::new(self.polytope.clone().normalize()))
    }

    fn distance<'py>(&self, _py: Python<'py>, point: PyArrayLike1<'py, f64>) -> PyResult<&'py PyArray1<f64>> {
        Ok(PyArray::from_array(_py, &self.polytope.distance(&point.into_owned_array())))
    }

    fn contains<'py>(&self, _py: Python<'py>, point: PyArrayLike1<'py, f64>) -> bool {
        self.polytope.contains(&point.into_owned_array())
    }

    fn translate<'py>(&self, _py: Python<'py>, point: PyArrayLike1<'py, f64>) -> PolytopeWrapper {
        PolytopeWrapper::new(self.polytope.translate(&point.into_owned_array()))
    }

    fn intersection(&self, other: &PolytopeWrapper) -> PolytopeWrapper {
        PolytopeWrapper::new(self.polytope.intersection(&other.polytope))
    }

    fn chebyshev_center<'py>(&self, _py: Python<'py>) -> PyResult<(PolytopeWrapper, &'py PyArray1<f64>)> {
        let (poly, cost) = self.polytope.chebyshev_center();
        Ok((PolytopeWrapper::new(poly), PyArray::from_array(_py, &cost)))
    }
    
    fn solve<'py>(&self, _py: Python<'py>, cost: Option<PyArrayLike1<'py, f64>>) -> PyResult<&'py PyArray1<f64>> {
        let cost = match cost {
            Some(val) => val.into_owned_array(),
            None => Array1::zeros(self.polytope.mat.len_of(Axis(1)))
        };

        let lp_state = self.polytope.solve_linprog(cost, false);

        match lp_state {
            PolytopeStatus::Optimal(solution) => Ok(PyArray::from_array(_py, &solution)),
            PolytopeStatus::Infeasible => Err(PyValueError::new_err("polytope is infeasible")),
            PolytopeStatus::Unbounded => Err(PyValueError::new_err("polytope is unbounded")),
            PolytopeStatus::Error(msg) => Err(PyValueError::new_err(msg)),
        }
    }

    fn __and__(&self, other: &PolytopeWrapper) -> PolytopeWrapper {
        self.intersection(other)
    }

    fn __repr__(&self) -> String {
        self.polytope.to_string()
    }

    fn __str__(&self) -> String {
        self.polytope.to_string()
    }

    // for backwards compatibility
    #[allow(non_snake_case)]
    fn to_Axbleqz<'py>(&self, _py: Python<'py>) -> PyResult<(&'py PyArray2<f64>, &'py PyArray1<f64>)> {
        Ok((PyArray::from_array(_py, &self.polytope.matrix_view()), PyArray::from_array(_py, &-&self.polytope.bias_view())))
    }

    #[allow(non_snake_case)]
    fn to_Axleqb<'py>(&self, _py: Python<'py>) -> PyResult<(&'py PyArray2<f64>, &'py PyArray1<f64>)> {
        Ok((PyArray::from_array(_py, &self.polytope.matrix_view()), PyArray::from_array(_py, &self.polytope.bias_view())))
    }

    #[allow(non_snake_case)]
    fn to_Axbgeqz<'py>(&self, _py: Python<'py>) -> PyResult<(&'py PyArray2<f64>, &'py PyArray1<f64>)> {
        Ok((PyArray::from_array(_py, &-&self.polytope.matrix_view()), PyArray::from_array(_py, &self.polytope.bias_view())))
    }

    #[allow(non_snake_case)]
    fn to_Axgeqb<'py>(&self, _py: Python<'py>) -> PyResult<(&'py PyArray2<f64>, &'py PyArray1<f64>)> {
        Ok((PyArray::from_array(_py, &-&self.polytope.matrix_view()), PyArray::from_array(_py, &-&self.polytope.bias_view())))
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "AffFunc")]
pub struct AffineFunctionWrapper {
    aff_func: AffFunc,
}

impl AffineFunctionWrapper {
    fn new(aff: AffFunc) -> AffineFunctionWrapper {
        AffineFunctionWrapper {
            aff_func: aff
        }
    }
}

#[pymethods]
impl AffineFunctionWrapper {
    #[staticmethod]
    fn from_mats<'py>(mat: PyArrayLike2<'py, f64>, bias: PyArrayLike1<'py, f64>) -> PyResult<AffineFunctionWrapper> {
        Ok(AffineFunctionWrapper::new(afffunc(mat, bias)?))
    }

    #[staticmethod]
    pub fn identity(dim: usize) -> AffineFunctionWrapper {
        AffineFunctionWrapper::new(AffFunc::identity(dim))
    }

    #[staticmethod]
    pub fn zeros(dim: usize) -> AffineFunctionWrapper {
        AffineFunctionWrapper::new(AffFunc::zeros(dim))
    }

    #[staticmethod]
    pub fn constant(dim: usize, value: f64) -> AffineFunctionWrapper {
        AffineFunctionWrapper::new(AffFunc::constant(dim, value))
    }

    #[staticmethod]
    pub fn unit(dim: usize, column: usize) -> AffineFunctionWrapper {
        AffineFunctionWrapper::new(AffFunc::unit(dim, column))
    }

    #[staticmethod]
    pub fn zero_idx(dim: usize, index: usize) -> AffineFunctionWrapper {
        AffineFunctionWrapper::new(AffFunc::zero_idx(dim, index))
    }

    #[staticmethod]
    pub fn rotation(orthogonal_mat: PyArrayLike2<'_, f64>) -> AffineFunctionWrapper {
        AffineFunctionWrapper::new(AffFunc::rotation(orthogonal_mat.into_owned_array()))
    }

    #[staticmethod]
    pub fn uniform_scaling(dim: usize, scalar: f64) -> AffineFunctionWrapper {
        AffineFunctionWrapper::new(AffFunc::uniform_scaling(dim, scalar))
    }

    #[staticmethod]
    pub fn scaling(scalars: PyArrayLike1<'_, f64>) -> AffineFunctionWrapper {
        AffineFunctionWrapper::new(AffFunc::scaling(&scalars.into_owned_array()))
    }

    #[staticmethod]
    pub fn slice(reference_point: PyArrayLike1<'_, f64>) -> AffineFunctionWrapper {
        AffineFunctionWrapper::new(AffFunc::slice(&reference_point.into_owned_array()))
    }

    #[staticmethod]
    pub fn translation(dim: usize, offset: PyArrayLike1<'_, f64>) -> AffineFunctionWrapper {
        AffineFunctionWrapper::new(AffFunc::translation(dim, offset.into_owned_array()))
    }

    #[getter]
    fn mat<'py>(&self, _py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        Ok(PyArray::from_array(_py, &self.aff_func.matrix_view()))
    }

    #[getter]
    fn bias<'py>(&self, _py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
        Ok(PyArray::from_array(_py, &self.aff_func.bias_view()))
    }

    fn indim(&self) -> PyResult<usize> {
        Ok(self.aff_func.indim())
    }

    fn outdim(&self) -> PyResult<usize> {
        Ok(self.aff_func.outdim())
    }

    fn row(&self, row: usize) -> PyResult<AffineFunctionWrapper> {
        Ok(AffineFunctionWrapper::new(self.aff_func.row(row).to_owned()))
    }

    fn row_iter(&self) -> PyResult<Vec<AffineFunctionWrapper>> {
        Ok(self.aff_func.row_iter().map(|x| AffineFunctionWrapper::new(x.to_owned())).collect())
    }

    pub fn apply<'py>(&self, _py: Python<'py>, input: PyArrayLike1<'py, f64>) -> PyResult<&'py PyArray1<f64>> {
        Ok(PyArray::from_array(_py, &self.aff_func.apply(&input.into_owned_array())))
    }

    pub fn apply_transpose<'py>(&self, _py: Python<'py>, input: PyArrayLike1<'py, f64>) -> PyResult<&'py PyArray1<f64>> {
        Ok(PyArray::from_array(_py, &self.aff_func.apply_transpose(&input.into_owned_array())))
    }

    pub fn compose(&self, other: &AffineFunctionWrapper) -> AffineFunctionWrapper {
        AffineFunctionWrapper::new(self.aff_func.compose(&other.aff_func))
    }

    pub fn stack(&self, other: &AffineFunctionWrapper) -> AffineFunctionWrapper {
        AffineFunctionWrapper::new(self.aff_func.stack(&other.aff_func))
    }

    pub fn __add__(&self, other: &AffineFunctionWrapper) -> AffineFunctionWrapper {
        AffineFunctionWrapper::new(&self.aff_func + &other.aff_func)
    }

    pub fn __sub__(&self, other: &AffineFunctionWrapper) -> AffineFunctionWrapper {
        AffineFunctionWrapper::new(&self.aff_func - &other.aff_func)
    }

    pub fn __mul__(&self, other: &AffineFunctionWrapper) -> AffineFunctionWrapper {
        AffineFunctionWrapper::new(&self.aff_func * &other.aff_func)
    }

    pub fn __div__(&self, other: &AffineFunctionWrapper) -> AffineFunctionWrapper {
        AffineFunctionWrapper::new(&self.aff_func / &other.aff_func)
    }

    pub fn __mod__(&self, other: &AffineFunctionWrapper) -> AffineFunctionWrapper {
        AffineFunctionWrapper::new(&self.aff_func % &other.aff_func)
    }

    pub fn __neg__(&self) -> AffineFunctionWrapper {
        AffineFunctionWrapper::new(-self.aff_func.clone())
    }

    fn __getitem__(&self, idx: SliceOrInt) -> PyResult<AffineFunctionWrapper> {
        match idx {
            SliceOrInt::Slice(slice) => {
                Err(PyNotImplementedError::new_err(""))
            }
            SliceOrInt::Int(index) => {
                Ok(AffineFunctionWrapper::new(self.aff_func.row(index as usize).to_owned()))
            }
        }
    }
}

#[pyclass]
#[derive(Clone)]
#[pyo3(name = "LayerBuilder")]
pub struct LayerBuilderW {
    pub layers: Vec<Layer>,
    pub input_dim: usize,
    pub current_dim: usize
}

#[pymethods]
impl LayerBuilderW {
    #[new]
    pub fn __init__(input_dim: usize) -> LayerBuilderW {
        LayerBuilderW { layers: Vec::new(), input_dim, current_dim: input_dim }
    }

    pub fn linear(&mut self, aff: AffineFunctionWrapper) {
        assert_eq!(self.current_dim, aff.aff_func.indim());
        self.current_dim = aff.aff_func.outdim();
        self.layers.push(Layer::Linear(aff.aff_func));
    }

    pub fn partial_relu(&mut self, idx: usize) {
        assert!(idx < self.current_dim);
        self.layers.push(Layer::ReLU(idx))
    }

    pub fn relu(&mut self) {
        for idx in 0..self.current_dim {
            self.partial_relu(idx)
        }
    }

    pub fn partial_leaky_relu(&mut self, idx: usize, alpha: f64) {
        assert!(idx < self.current_dim);
        self.layers.push(Layer::LeakyReLU(idx, alpha))
    }

    pub fn leaky_relu(&mut self, alpha: f64) {
        for idx in 0..self.current_dim {
            self.partial_leaky_relu(idx, alpha)
        }
    }

    pub fn partial_hard_tanh(&mut self, idx: usize) {
        assert!(idx < self.current_dim);
        self.layers.push(Layer::HardTanh(idx))
    }

    pub fn hard_tanh(&mut self) {
        for idx in 0..self.current_dim {
            self.partial_hard_tanh(idx)
        }
    }

    pub fn partial_hard_sigmoid(&mut self, idx: usize) {
        assert!(idx < self.current_dim);
        self.layers.push(Layer::HardSigmoid(idx))
    }

    pub fn hard_sigmoid(&mut self) {
        for idx in 0..self.current_dim {
            self.partial_hard_sigmoid(idx)
        }
    }

    pub fn argmax(&mut self) {
        self.current_dim = 1;
        self.layers.push(Layer::Argmax)
    }

    pub fn __str__(&self) -> String {
        use std::fmt::Write;

        let mut str = String::new();
        for layer in &self.layers {
            match layer {
                Layer::Linear(aff) => {
                    writeln!(&mut str, "- linear {}x{}", aff.indim(), aff.outdim());
                }
                Layer::ReLU(idx) => {
                    writeln!(&mut str, "- ReLU idx={}", idx);
                }
                Layer::LeakyReLU(idx, alpha) => {
                    writeln!(&mut str, "- LeakyReLU alpha={} idx={}", alpha, idx);
                }
                Layer::HardTanh(idx) => {
                    writeln!(&mut str, "- HardTanh idx={}", idx);
                }
                Layer::HardSigmoid(idx) => {
                    writeln!(&mut str, "- HardSigmoid idx={}", idx);
                }
                Layer::Argmax => {
                    writeln!(&mut str, "- Argmax");
                }
                Layer::ClassChar(class) => {
                    writeln!(&mut str, "- ClassChar class={}", class);
                }
            }
        }
        str
    }
}

impl IntoIterator for LayerBuilderW {
    type Item = Layer;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.layers.into_iter()
    }
}

#[allow(non_snake_case)]
#[pyfunction]
fn ReLU(dim: usize) -> PyResult<AffTreeWrapper> {
    Ok(AffTreeWrapper { aff_tree: schema::ReLU(dim) })
}

#[allow(non_snake_case)]
#[pyfunction]
fn partial_ReLU(dim: usize, row: usize) -> PyResult<AffTreeWrapper> {
    Ok(AffTreeWrapper { aff_tree: schema::partial_ReLU(dim, row) })
}

#[allow(non_snake_case)]
#[pyfunction]
fn partial_leaky_ReLU(dim: usize, row: usize, alpha: f64) -> PyResult<AffTreeWrapper> {
    Ok(AffTreeWrapper { aff_tree: schema::partial_leaky_ReLU(dim, row, alpha) })
}


#[allow(non_snake_case)]
#[pyfunction]
fn partial_hard_tanh(dim: usize, row: usize, min_val: f64, max_val: f64) -> PyResult<AffTreeWrapper> {
    Ok(AffTreeWrapper { aff_tree: schema::partial_hard_tanh(dim, row, min_val, max_val) })
}

#[allow(non_snake_case)]
#[pyfunction]
fn partial_hard_sigmoid(dim: usize, row: usize) -> PyResult<AffTreeWrapper> {
    Ok(AffTreeWrapper { aff_tree: schema::partial_hard_sigmoid(dim, row) })
}

#[pyfunction]
fn argmax(dim: usize) -> PyResult<AffTreeWrapper> {
    Ok(AffTreeWrapper { aff_tree: schema::argmax(dim) })
}

#[pyfunction]
fn class_characterization(dim: usize, clazz: usize) -> PyResult<AffTreeWrapper> {
    Ok(AffTreeWrapper { aff_tree: schema::class_characterization(dim, clazz) })
}

#[pyfunction]
fn inf_norm(dim: usize, minimum: Option<f64>, maximum: Option<f64>) -> PyResult<AffTreeWrapper> {
    Ok(AffTreeWrapper { aff_tree: schema::inf_norm(dim, minimum, maximum) })
}

#[pymodule]
fn rust_schema(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "schema")?;
    child_module.add_function(wrap_pyfunction!(ReLU, m)?)?;
    child_module.add_function(wrap_pyfunction!(partial_ReLU, m)?)?;
    child_module.add_function(wrap_pyfunction!(argmax, m)?)?;
    child_module.add_function(wrap_pyfunction!(class_characterization, m)?)?;
    child_module.add_function(wrap_pyfunction!(inf_norm, m)?)?;
    m.add_submodule(child_module);
    Ok(())
}


#[pyfunction]
fn from_layers(layers: LayerBuilderW, precondition: Option<AffTreeWrapper>, csv: Option<&str>) -> PyResult<AffTreeWrapper> {
    let dim = if let Some(ref pre) = precondition {
        pre.aff_tree.in_dim()
    } else {
        layers.input_dim
    };

    let aff_tree = if let Some(path) = csv {
        afftree_from_layers_csv(dim, layers, path, precondition.map(|x| x.aff_tree))
    } else {
        afftree_from_layers_verbose(dim, layers, precondition.map(|x| x.aff_tree))
    };
        
    Ok(AffTreeWrapper { aff_tree })
}

#[pyfunction]
fn read_npz(dim: usize, filename: String, precondition: Option<AffTreeWrapper>, csv: Option<&str>) -> PyResult<AffTreeWrapper> {
    let layers = match read_layers(&filename) {
        Ok(layers) => layers,
        Err(e) => return Err(PyValueError::new_err("Error reading numpy file"))
    };

    let aff_tree = if let Some(path) = csv {
        afftree_from_layers_csv(dim, &layers, path, precondition.map(|x| x.aff_tree))
    } else {
        afftree_from_layers_verbose(dim, &layers, precondition.map(|x| x.aff_tree))
    };
    
    Ok(AffTreeWrapper { aff_tree })
}

#[pymodule]
fn rust_builder(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    let child_module = PyModule::new(py, "builder")?;
    child_module.add_function(wrap_pyfunction!(from_layers, m)?)?;
    child_module.add_function(wrap_pyfunction!(read_npz, m)?)?;
    m.add_submodule(child_module);
    Ok(())
}

#[pymodule]
fn affinitree(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    rust_schema(py, m);
    rust_builder(py, m);
    m.add_class::<AffTreeWrapper>()?;
    m.add_class::<AffineNodeWrapper>()?;
    m.add_class::<PolytopeWrapper>()?;
    m.add_class::<AffineFunctionWrapper>()?;
    m.add_class::<LayerBuilderW>()?;
    Ok(())

}