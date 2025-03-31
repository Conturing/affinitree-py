//   Copyright 2025 affinitree developers
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

use std::error;

use affinitree_rust::distill::arch::{Architecture, ShapeError, TensorShape};
use pyo3::exceptions::{PyException, PyIOError, PyNotImplementedError, PyValueError};
use pyo3::panic::PanicException;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PySlice, PyType};

use numpy::ndarray::{Array1, Array2, Axis};
use numpy::{
    IntoPyArray, Ix1, PyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray, PyReadonlyArray1,
    PyReadonlyArray2, PyUntypedArray,
};

use pyo3_arraylike::{PyArrayLike1, PyArrayLike2};

use affinitree_rust::distill::builder::{
    self, afftree_from_layers, afftree_from_layers_csv, afftree_from_layers_verbose, read_layers,
    Layer,
};
use affinitree_rust::distill::schema;
use affinitree_rust::linalg::affine::{AffFunc, Polytope};
use affinitree_rust::linalg::polyhedron::PolytopeStatus;
use affinitree_rust::pwl::afftree::{AffTree, InputError};
use affinitree_rust::pwl::dot::{self, Dot};
use affinitree_rust::pwl::node::{AffContent, AffNode};
use affinitree_rust::tree::graph::{
    InvalidTreeIndexError, Label, NodeError, Tree, TreeIndex, TreeNode,
};

use thiserror::Error;

#[derive(FromPyObject)]
pub enum SliceOrInt<'a> {
    Slice(&'a PySlice),
    Int(isize),
}

fn afffunc<'py>(
    mat: PyArrayLike2<'py, f64>,
    bias: PyArrayLike1<'py, f64>,
) -> Result<AffFunc, PyErr> {
    let mat = mat.into_owned_array();
    let bias = bias.into_owned_array();
    if mat.shape()[0] != bias.shape()[0] {
        Err(PyValueError::new_err(format!(
            "Dimensions mismatch: {} vs {}",
            mat.shape()[0],
            bias.shape()[0]
        )))
    } else {
        Ok(AffFunc::from_mats(mat, bias))
    }
}

#[derive(Error, Debug)]
pub enum AffinitreeError {
    #[error(transparent)]
    NodeError(#[from] NodeError),

    #[error(transparent)]
    InvalidTreeIndexError(#[from] InvalidTreeIndexError),

    #[error(transparent)]
    InputError(#[from] InputError),

    #[error(transparent)]
    ShapeError(#[from] ShapeError),
}

impl From<AffinitreeError> for PyErr {
    fn from(err: AffinitreeError) -> PyErr {
        match err {
            AffinitreeError::NodeError(e) => PyValueError::new_err(format!("{}", e)),
            AffinitreeError::InvalidTreeIndexError(e) => PyValueError::new_err(format!("{}", e)),
            AffinitreeError::InputError(e) => PyValueError::new_err(format!("{}", e)),
            AffinitreeError::ShapeError(e) => PyValueError::new_err(format!("{}", e)),
        }
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "AffTree")]
struct PyAffTree {
    aff_tree: AffTree<2>,
}

#[pymethods]
impl PyAffTree {
    #[staticmethod]
    fn identity(dim: usize) -> PyResult<PyAffTree> {
        Ok(PyAffTree {
            aff_tree: AffTree::from_aff(AffFunc::identity(dim)),
        })
    }

    #[staticmethod]
    fn from_aff(func: &PyAffFunc) -> PyResult<PyAffTree> {
        Ok(PyAffTree {
            aff_tree: AffTree::from_aff(func.aff_func.clone()),
        })
    }

    #[staticmethod]
    fn from_array<'py>(
        weights: PyArrayLike2<'py, f64>,
        bias: PyArrayLike1<'py, f64>,
    ) -> PyResult<PyAffTree> {
        Ok(PyAffTree {
            aff_tree: AffTree::from_aff(afffunc(weights, bias)?),
        })
    }

    #[staticmethod]
    fn from_poly(
        precondition: PyPolytope,
        func_true: PyAffFunc,
        func_false: Option<PyAffFunc>,
    ) -> PyResult<PyAffTree> {
        let tree = AffTree::from_poly(
            precondition.polytope,
            func_true.aff_func,
            func_false.map(|w| w.aff_func).as_ref(),
        );
        match tree {
            Ok(aff_tree) => Ok(PyAffTree { aff_tree }),
            Err(e) => Err(PyErr::from(AffinitreeError::from(e))),
        }
    }

    /* Status */

    fn indim(&self) -> PyResult<usize> {
        Ok(self.aff_tree.in_dim)
    }

    fn size(&self) -> PyResult<usize> {
        Ok(self.aff_tree.len())
    }

    fn is_empty(&self) -> bool {
        self.aff_tree.is_empty()
    }

    fn num_terminals(&self) -> PyResult<usize> {
        Ok(self.aff_tree.num_terminals())
    }

    fn depth(&self) -> PyResult<usize> {
        Ok(self.aff_tree.tree.depth())
    }

    fn reserve(&mut self, additional: usize) {
        self.aff_tree.reserve(additional);
    }

    /* Iterators */

    fn nodes(&self) -> PyResult<Vec<AffineNodeWrapper>> {
        Ok(self
            .aff_tree
            .tree
            .node_iter()
            .map(|(idx, node)| AffineNodeWrapper {
                node: node.clone(),
                idx,
            })
            .collect())
    }

    fn terminals(&self) -> PyResult<Vec<AffineNodeWrapper>> {
        Ok(self
            .aff_tree
            .tree
            .terminals()
            .map(|nd| {
                let node = self.aff_tree.tree.tree_node(nd.idx).unwrap();
                AffineNodeWrapper::new(nd.idx, node.clone())
            })
            .collect())
    }

    fn decisions(&self) -> PyResult<Vec<AffineNodeWrapper>> {
        Ok(self
            .aff_tree
            .tree
            .decisions()
            .map(|nd| {
                let node = self.aff_tree.tree.tree_node(nd.idx).unwrap();
                AffineNodeWrapper::new(nd.idx, node.clone())
            })
            .collect())
    }

    fn polyhedra(&self) -> PyResult<Vec<(usize, AffineNodeWrapper, PyPolytope)>> {
        Ok(self
            .aff_tree
            .polyhedra_iter()
            .map(|(depth, idx, n_remaining, poly)| {
                (
                    depth,
                    AffineNodeWrapper::new(idx, self.aff_tree.tree.tree_node(idx).unwrap().clone()),
                    PyPolytope {
                        polytope: Polytope::intersection_n(self.aff_tree.in_dim(), poly.as_slice()),
                    },
                )
            })
            .collect())
    }

    fn dfs(&self) -> PyResult<Vec<(usize, AffineNodeWrapper, usize)>> {
        Ok(self
            .aff_tree
            .tree
            .dfs_iter()
            .map(|data| {
                (
                    data.depth,
                    AffineNodeWrapper::new(
                        data.index,
                        self.aff_tree.tree.tree_node(data.index).unwrap().clone(),
                    ),
                    data.n_remaining,
                )
            })
            .collect())
    }

    fn edges(&self) -> PyResult<Vec<(AffineNodeWrapper, Label, AffineNodeWrapper)>> {
        Ok(self
            .aff_tree
            .tree
            .edge_iter()
            .map(|edg| {
                let src_node = self.aff_tree.tree.tree_node(edg.source_idx).unwrap();
                let tgt_node = self.aff_tree.tree.tree_node(edg.target_idx).unwrap();
                (
                    AffineNodeWrapper::new(edg.source_idx, src_node.clone()),
                    edg.label,
                    AffineNodeWrapper::new(edg.target_idx, tgt_node.clone()),
                )
            })
            .collect())
    }

    pub fn add_child_node(
        &mut self,
        parent: AffineNodeWrapper,
        label: Label,
        aff: PyAffFunc,
    ) -> PyResult<AffineNodeWrapper> {
        let idx = self
            .aff_tree
            .add_child_node(parent.idx, label, aff.aff_func);
        match idx {
            Ok(idx) => Ok(AffineNodeWrapper::new(
                idx,
                self.aff_tree.tree.tree_node(idx).unwrap().clone(),
            )),
            Err(e) => Err(PyErr::from(AffinitreeError::from(e))),
        }
    }

    pub fn update_node(&mut self, node: AffineNodeWrapper, aff: PyAffFunc) -> PyResult<PyAffFunc> {
        self.aff_tree
            .update_node(node.idx, aff.aff_func)
            .map(|val| PyAffFunc { aff_func: val })
            .map_err(|e| PyErr::from(AffinitreeError::from(e)))
    }

    fn apply_func(&mut self, aff_func: &PyAffFunc) {
        self.aff_tree.apply_func(&aff_func.aff_func);
    }

    fn evaluate<'py>(
        &self,
        _py: Python<'py>,
        input: PyArrayLike1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let res = self.aff_tree.evaluate(&input.into_owned_array());
        match res {
            Some(val) => Ok(PyArray::from_array_bound(_py, &val)),
            None => Err(PyValueError::new_err(
                "Evaluation failed: node as no child for given input",
            )),
        }
    }

    fn compose(&mut self, other: &PyAffTree, prune: Option<bool>) {
        if let Some(false) = prune {
            self.aff_tree.compose::<false, false>(&other.aff_tree);
        } else {
            self.aff_tree.compose::<true, false>(&other.aff_tree);
        }
    }

    fn infeasible_elimination(&mut self) {
        self.aff_tree.infeasible_elimination();
    }

    fn reduce(&mut self) {
        self.aff_tree.reduce()
    }

    #[getter]
    fn root(&self) -> PyResult<AffineNodeWrapper> {
        Ok(AffineNodeWrapper::new(
            self.aff_tree.tree.get_root_idx(),
            self.aff_tree.tree.get_root().clone(),
        ))
    }

    fn parent(&self, node: &AffineNodeWrapper) -> PyResult<AffineNodeWrapper> {
        let res = self.aff_tree.tree.parent(node.idx);
        match res {
            Ok(edg) => Ok(AffineNodeWrapper::new(
                edg.source_idx,
                self.aff_tree
                    .tree
                    .tree_node(edg.source_idx)
                    .unwrap()
                    .clone(),
            )),
            Err(e) => Err(PyErr::from(AffinitreeError::from(e))),
        }
    }

    fn child(&self, node: &AffineNodeWrapper, label: usize) -> PyResult<AffineNodeWrapper> {
        let res = self.aff_tree.tree.child(node.idx, label);
        match res {
            Ok(edg) => Ok(AffineNodeWrapper::new(
                edg.target_idx,
                self.aff_tree
                    .tree
                    .tree_node(edg.target_idx)
                    .unwrap()
                    .clone(),
            )),
            Err(e) => Err(PyErr::from(AffinitreeError::from(e))),
        }
    }

    fn remove_axes<'py>(&mut self, _py: Python<'py>, mask: PyArrayLike1<'py, bool>) {
        self.aff_tree.remove_axes(&mask.into_owned_array()).unwrap()
    }

    fn to_dot(&self) -> String {
        let dot = Dot::from(&self.aff_tree);
        dot.to_string()
    }

    pub fn __neg__(&self) -> PyAffTree {
        PyAffTree {
            aff_tree: -self.aff_tree.clone(),
        }
    }

    pub fn __add__(&self, other: &PyAffTree) -> PyAffTree {
        PyAffTree {
            aff_tree: &self.aff_tree + &other.aff_tree,
        }
    }

    pub fn __sub__(&self, other: &PyAffTree) -> PyAffTree {
        PyAffTree {
            aff_tree: &self.aff_tree - &other.aff_tree,
        }
    }

    pub fn __mul__(&self, other: &PyAffTree) -> PyAffTree {
        PyAffTree {
            aff_tree: &self.aff_tree * &other.aff_tree,
        }
    }

    pub fn __div__(&self, other: &PyAffTree) -> PyAffTree {
        PyAffTree {
            aff_tree: &self.aff_tree / &other.aff_tree,
        }
    }

    pub fn __len__(&self) -> PyResult<usize> {
        self.size()
    }

    fn __getitem__(&self, key: TreeIndex) -> PyResult<AffineNodeWrapper> {
        match self.aff_tree.tree.tree_node(key) {
            Ok(x) => Ok(AffineNodeWrapper::new(key, x.clone())),
            Err(e) => Err(PyErr::from(AffinitreeError::from(e))),
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
    idx: TreeIndex,
}

impl AffineNodeWrapper {
    fn new(idx: TreeIndex, node: AffNode<2>) -> AffineNodeWrapper {
        AffineNodeWrapper { node, idx }
    }
}

#[pymethods]
impl AffineNodeWrapper {
    #[getter]
    fn val(&self) -> PyResult<PyAffFunc> {
        Ok(PyAffFunc {
            aff_func: self.node.value.aff.clone(),
        })
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
            _ => Err(PyNotImplementedError::new_err("")),
        }
    }

    fn __hash__(&self) -> usize {
        self.idx
    }

    fn __str__(&self) -> String {
        self.node.to_string()
    }

    fn __repr__(&self) -> String {
        format!("({:>4}, {})", self.idx, self.node)
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "Polytope")]
struct PyPolytope {
    polytope: Polytope,
}

impl PyPolytope {
    fn new(poly: Polytope) -> PyPolytope {
        PyPolytope { polytope: poly }
    }
}

#[pymethods]
impl PyPolytope {
    #[staticmethod]
    fn from_mats<'py>(
        weights: PyArrayLike2<'py, f64>,
        bias: PyArrayLike1<'py, f64>,
    ) -> PyResult<PyPolytope> {
        let mat = weights.into_owned_array();
        let bias = bias.into_owned_array();
        if mat.shape()[0] != bias.shape()[0] {
            Err(PyValueError::new_err(format!(
                "Dimensions mismatch: {} vs {}",
                mat.shape()[0],
                bias.shape()[0]
            )))
        } else {
            Ok(PyPolytope::new(Polytope::from_mats(mat, bias)))
        }
    }

    #[staticmethod]
    fn hyperrectangle(dim: usize, intervals: Vec<(f64, f64)>) -> PyResult<PyPolytope> {
        let polytope = Polytope::hyperrectangle(intervals.as_slice());
        Ok(PyPolytope::new(polytope))
    }

    fn indim(&self) -> PyResult<usize> {
        Ok(self.polytope.indim())
    }

    fn n_constraints(&self) -> PyResult<usize> {
        Ok(self.polytope.outdim())
    }

    #[getter]
    fn mat<'py>(&self, _py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(PyArray::from_array_bound(_py, &self.polytope.matrix_view()))
    }

    #[getter]
    fn bias<'py>(&self, _py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(PyArray::from_array_bound(_py, &self.polytope.bias_view()))
    }

    fn row(&self, row: usize) -> PyResult<PyPolytope> {
        Ok(PyPolytope::new(self.polytope.row(row).to_owned()))
    }

    fn row_iter(&self) -> PyResult<Vec<PyPolytope>> {
        Ok(self
            .polytope
            .row_iter()
            .map(|x| PyPolytope::new(x.to_owned()))
            .collect())
    }

    fn normalize(&self) -> PyResult<PyPolytope> {
        Ok(PyPolytope::new(self.polytope.clone().normalize()))
    }

    fn distance<'py>(
        &self,
        _py: Python<'py>,
        point: PyArrayLike1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(PyArray::from_array_bound(
            _py,
            &self.polytope.distance(&point.into_owned_array()),
        ))
    }

    fn contains<'py>(&self, _py: Python<'py>, point: PyArrayLike1<'py, f64>) -> bool {
        self.polytope.contains(&point.into_owned_array())
    }

    fn translate<'py>(&self, _py: Python<'py>, point: PyArrayLike1<'py, f64>) -> PyPolytope {
        PyPolytope::new(self.polytope.translate(&point.into_owned_array()))
    }

    fn intersection(&self, other: &PyPolytope) -> PyPolytope {
        PyPolytope::new(self.polytope.intersection(&other.polytope))
    }

    fn rotate<'py>(
        &self,
        _py: Python<'py>,
        orthogonal_matrix: PyArrayLike2<'py, f64>,
    ) -> PyPolytope {
        PyPolytope::new(self.polytope.rotate(&orthogonal_matrix.into_owned_array()))
    }

    fn chebyshev_center<'py>(
        &self,
        _py: Python<'py>,
    ) -> PyResult<(PyPolytope, Bound<'py, PyArray1<f64>>)> {
        let (poly, cost) = self.polytope.chebyshev_center();
        Ok((PyPolytope::new(poly), PyArray::from_array_bound(_py, &cost)))
    }

    fn solve<'py>(
        &self,
        _py: Python<'py>,
        cost: Option<PyArrayLike1<'py, f64>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let cost = match cost {
            Some(val) => val.into_owned_array(),
            None => Array1::zeros(self.polytope.mat.len_of(Axis(1))),
        };

        let lp_state = self.polytope.solve_linprog(cost, false);

        match lp_state {
            PolytopeStatus::Optimal(solution) => Ok(PyArray::from_array_bound(_py, &solution)),
            PolytopeStatus::Infeasible => Err(PyValueError::new_err("polytope is infeasible")),
            PolytopeStatus::Unbounded => Err(PyValueError::new_err("polytope is unbounded")),
            PolytopeStatus::Error(msg) => Err(PyValueError::new_err(msg)),
        }
    }

    fn __in__<'py>(&self, _py: Python<'py>, point: PyArrayLike1<'py, f64>) -> bool {
        self.contains(_py, point)
    }

    fn __and__(&self, other: &PyPolytope) -> PyPolytope {
        self.intersection(other)
    }

    fn __str__(&self) -> String {
        self.polytope.to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.polytope)
    }

    // for backwards compatibility
    #[allow(non_snake_case)]
    fn to_Axbleqz<'py>(
        &self,
        _py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
        Ok((
            PyArray::from_array_bound(_py, &self.polytope.matrix_view()),
            PyArray::from_array_bound(_py, &-&self.polytope.bias_view()),
        ))
    }

    #[allow(non_snake_case)]
    fn to_Axleqb<'py>(
        &self,
        _py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
        Ok((
            PyArray::from_array_bound(_py, &self.polytope.matrix_view()),
            PyArray::from_array_bound(_py, &self.polytope.bias_view()),
        ))
    }

    #[allow(non_snake_case)]
    fn to_Axbgeqz<'py>(
        &self,
        _py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
        Ok((
            PyArray::from_array_bound(_py, &-&self.polytope.matrix_view()),
            PyArray::from_array_bound(_py, &self.polytope.bias_view()),
        ))
    }

    #[allow(non_snake_case)]
    fn to_Axgeqb<'py>(
        &self,
        _py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
        Ok((
            PyArray::from_array_bound(_py, &-&self.polytope.matrix_view()),
            PyArray::from_array_bound(_py, &-&self.polytope.bias_view()),
        ))
    }
}

#[derive(Clone)]
#[pyclass]
#[pyo3(name = "AffFunc")]
pub struct PyAffFunc {
    aff_func: AffFunc,
}

impl PyAffFunc {
    fn new(aff: AffFunc) -> PyAffFunc {
        PyAffFunc { aff_func: aff }
    }
}

#[pymethods]
impl PyAffFunc {
    #[staticmethod]
    fn from_mats<'py>(
        mat: PyArrayLike2<'py, f64>,
        bias: PyArrayLike1<'py, f64>,
    ) -> PyResult<PyAffFunc> {
        Ok(PyAffFunc::new(afffunc(mat, bias)?))
    }

    #[staticmethod]
    pub fn identity(dim: usize) -> PyAffFunc {
        PyAffFunc::new(AffFunc::identity(dim))
    }

    #[staticmethod]
    pub fn zeros(dim: usize) -> PyAffFunc {
        PyAffFunc::new(AffFunc::zeros(dim))
    }

    #[staticmethod]
    pub fn constant(dim: usize, value: f64) -> PyAffFunc {
        PyAffFunc::new(AffFunc::constant(dim, value))
    }

    #[staticmethod]
    pub fn unit(dim: usize, index: usize) -> PyAffFunc {
        PyAffFunc::new(AffFunc::unit(dim, index))
    }

    #[staticmethod]
    pub fn zero_idx(dim: usize, index: usize) -> PyAffFunc {
        PyAffFunc::new(AffFunc::zero_idx(dim, index))
    }

    #[staticmethod]
    pub fn rotation(orthogonal_mat: PyArrayLike2<'_, f64>) -> PyAffFunc {
        PyAffFunc::new(AffFunc::rotation(orthogonal_mat.into_owned_array()))
    }

    #[staticmethod]
    pub fn uniform_scaling(dim: usize, scalar: f64) -> PyAffFunc {
        PyAffFunc::new(AffFunc::uniform_scaling(dim, scalar))
    }

    #[staticmethod]
    pub fn scaling(scalars: PyArrayLike1<'_, f64>) -> PyAffFunc {
        PyAffFunc::new(AffFunc::scaling(&scalars.into_owned_array()))
    }

    #[staticmethod]
    pub fn slice(reference_point: PyArrayLike1<'_, f64>) -> PyAffFunc {
        PyAffFunc::new(AffFunc::slice(&reference_point.into_owned_array()))
    }

    #[staticmethod]
    pub fn translation(dim: usize, offset: PyArrayLike1<'_, f64>) -> PyAffFunc {
        PyAffFunc::new(AffFunc::translation(dim, offset.into_owned_array()))
    }

    #[getter]
    fn mat<'py>(&self, _py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(PyArray::from_array_bound(_py, &self.aff_func.matrix_view()))
    }

    #[getter]
    fn bias<'py>(&self, _py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(PyArray::from_array_bound(_py, &self.aff_func.bias_view()))
    }

    fn indim(&self) -> PyResult<usize> {
        Ok(self.aff_func.indim())
    }

    fn outdim(&self) -> PyResult<usize> {
        Ok(self.aff_func.outdim())
    }

    fn row(&self, row: usize) -> PyResult<PyAffFunc> {
        Ok(PyAffFunc::new(self.aff_func.row(row).to_owned()))
    }

    fn row_iter(&self) -> PyResult<Vec<PyAffFunc>> {
        Ok(self
            .aff_func
            .row_iter()
            .map(|x| PyAffFunc::new(x.to_owned()))
            .collect())
    }

    pub fn apply<'py>(
        &self,
        _py: Python<'py>,
        input: PyArrayLike1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(PyArray::from_array_bound(
            _py,
            &self.aff_func.apply(&input.into_owned_array()),
        ))
    }

    pub fn apply_transpose<'py>(
        &self,
        _py: Python<'py>,
        input: PyArrayLike1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(PyArray::from_array_bound(
            _py,
            &self.aff_func.apply_transpose(&input.into_owned_array()),
        ))
    }

    pub fn compose(&self, other: &PyAffFunc) -> PyAffFunc {
        PyAffFunc::new(self.aff_func.compose(&other.aff_func))
    }

    pub fn stack(&self, other: &PyAffFunc) -> PyAffFunc {
        PyAffFunc::new(self.aff_func.stack(&other.aff_func))
    }

    pub fn __call__<'py>(
        &self,
        _py: Python<'py>,
        input: PyArrayLike1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.apply(_py, input)
    }

    pub fn __add__(&self, other: &PyAffFunc) -> PyAffFunc {
        PyAffFunc::new(&self.aff_func + &other.aff_func)
    }

    pub fn __sub__(&self, other: &PyAffFunc) -> PyAffFunc {
        PyAffFunc::new(&self.aff_func - &other.aff_func)
    }

    pub fn __mul__(&self, other: &PyAffFunc) -> PyAffFunc {
        PyAffFunc::new(&self.aff_func * &other.aff_func)
    }

    pub fn __div__(&self, other: &PyAffFunc) -> PyAffFunc {
        PyAffFunc::new(&self.aff_func / &other.aff_func)
    }

    pub fn __mod__(&self, other: &PyAffFunc) -> PyAffFunc {
        PyAffFunc::new(&self.aff_func % &other.aff_func)
    }

    pub fn __neg__(&self) -> PyAffFunc {
        PyAffFunc::new(-self.aff_func.clone())
    }

    fn __getitem__(&self, idx: SliceOrInt) -> PyResult<PyAffFunc> {
        match idx {
            SliceOrInt::Slice(slice) => Err(PyNotImplementedError::new_err("")),
            SliceOrInt::Int(index) => {
                Ok(PyAffFunc::new(self.aff_func.row(index as usize).to_owned()))
            }
        }
    }

    fn __str__(&self) -> String {
        self.aff_func.to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.aff_func)
    }
}

#[pyclass]
#[derive(Clone)]
#[pyo3(name = "Architecture")]
pub struct PyArchitecture {
    pub arch: Architecture,
}

#[pymethods]
impl PyArchitecture {
    #[new]
    pub fn __init__(input_dim: usize) -> PyArchitecture {
        PyArchitecture {
            arch: Architecture::new(TensorShape::Flat { in_dim: input_dim }),
        }
    }

    pub fn linear(&mut self, aff: PyAffFunc) -> PyResult<()> {
        let rv = self.arch.linear(aff.aff_func);
        rv.map_err(|e| PyErr::from(AffinitreeError::from(e)))
    }

    pub fn partial_relu(&mut self, idx: usize) -> PyResult<()> {
        let rv = self.arch.partial_relu(idx);
        rv.map_err(|e| PyErr::from(AffinitreeError::from(e)))
    }

    pub fn relu(&mut self) -> PyResult<()> {
        let rv = self.arch.relu();
        rv.map_err(|e| PyErr::from(AffinitreeError::from(e)))
    }

    pub fn partial_leaky_relu(&mut self, idx: usize, alpha: f64) -> PyResult<()> {
        let rv = self.arch.partial_leaky_relu(idx, alpha);
        rv.map_err(|e| PyErr::from(AffinitreeError::from(e)))
    }

    pub fn leaky_relu(&mut self, alpha: f64) -> PyResult<()> {
        let rv = self.arch.leaky_relu(alpha);
        rv.map_err(|e| PyErr::from(AffinitreeError::from(e)))
    }

    pub fn partial_hard_tanh(&mut self, idx: usize) -> PyResult<()> {
        let rv = self.arch.partial_hard_tanh(idx);
        rv.map_err(|e| PyErr::from(AffinitreeError::from(e)))
    }

    pub fn hard_tanh(&mut self) -> PyResult<()> {
        let rv = self.arch.hard_tanh();
        rv.map_err(|e| PyErr::from(AffinitreeError::from(e)))
    }

    pub fn partial_hard_sigmoid(&mut self, idx: usize) -> PyResult<()> {
        let rv = self.arch.partial_hard_sigmoid(idx);
        rv.map_err(|e| PyErr::from(AffinitreeError::from(e)))
    }

    pub fn hard_sigmoid(&mut self) -> PyResult<()> {
        let rv = self.arch.hard_sigmoid();
        rv.map_err(|e| PyErr::from(AffinitreeError::from(e)))
    }

    pub fn argmax(&mut self) -> PyResult<()> {
        let rv = self.arch.argmax();
        rv.map_err(|e| PyErr::from(AffinitreeError::from(e)))
    }

    pub fn extract_range(&self, start: usize, end: usize) -> PyResult<Self> {
        self.arch
            .extract_range(start, end)
            .map(|sub_arch| PyArchitecture { arch: sub_arch })
            .map_err(|e| PyErr::from(AffinitreeError::from(e)))
    }

    pub fn __len__(&self) -> usize {
        self.arch.operators.len()
    }

    pub fn __str__(&self) -> String {
        self.arch.to_string()
    }
}

impl IntoIterator for PyArchitecture {
    type Item = Layer;
    type IntoIter = <Architecture as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.arch.into_iter()
    }
}

#[allow(non_snake_case)]
#[pyfunction]
fn partial_ReLU(dim: usize, row: usize) -> PyResult<PyAffTree> {
    Ok(PyAffTree {
        aff_tree: schema::partial_ReLU(dim, row),
    })
}

#[allow(non_snake_case)]
#[pyfunction]
fn partial_leaky_ReLU(dim: usize, row: usize, alpha: f64) -> PyResult<PyAffTree> {
    Ok(PyAffTree {
        aff_tree: schema::partial_leaky_ReLU(dim, row, alpha),
    })
}

#[allow(non_snake_case)]
#[pyfunction]
fn partial_hard_tanh(dim: usize, row: usize, min_val: f64, max_val: f64) -> PyResult<PyAffTree> {
    Ok(PyAffTree {
        aff_tree: schema::partial_hard_tanh(dim, row, min_val, max_val),
    })
}

#[allow(non_snake_case)]
#[pyfunction]
fn partial_hard_sigmoid(dim: usize, row: usize) -> PyResult<PyAffTree> {
    Ok(PyAffTree {
        aff_tree: schema::partial_hard_sigmoid(dim, row),
    })
}

#[pyfunction]
fn argmax(dim: usize) -> PyResult<PyAffTree> {
    Ok(PyAffTree {
        aff_tree: schema::argmax(dim),
    })
}

#[pyfunction]
fn class_characterization(dim: usize, clazz: usize) -> PyResult<PyAffTree> {
    Ok(PyAffTree {
        aff_tree: schema::class_characterization(dim, clazz),
    })
}

#[pyfunction]
fn inf_norm(dim: usize, minimum: Option<f64>, maximum: Option<f64>) -> PyResult<PyAffTree> {
    Ok(PyAffTree {
        aff_tree: schema::inf_norm(dim, minimum, maximum),
    })
}

#[pymodule]
fn rust_schema(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let child_module = PyModule::new_bound(m.py(), "schema")?;
    child_module.add_function(wrap_pyfunction!(partial_ReLU, m)?)?;
    child_module.add_function(wrap_pyfunction!(partial_leaky_ReLU, m)?)?;
    child_module.add_function(wrap_pyfunction!(partial_hard_tanh, m)?)?;
    child_module.add_function(wrap_pyfunction!(partial_hard_sigmoid, m)?)?;
    child_module.add_function(wrap_pyfunction!(argmax, m)?)?;
    child_module.add_function(wrap_pyfunction!(class_characterization, m)?)?;
    child_module.add_function(wrap_pyfunction!(inf_norm, m)?)?;
    m.add_submodule(&child_module);
    Ok(())
}

#[pyfunction]
fn from_layers(
    layers: PyArchitecture,
    precondition: Option<PyAffTree>,
    csv: Option<&str>,
) -> PyResult<PyAffTree> {
    let dim = if let Some(ref pre) = precondition {
        pre.aff_tree.in_dim()
    } else {
        layers.arch.input_shape.max_dim()
    };

    let aff_tree = if let Some(path) = csv {
        afftree_from_layers_csv(dim, layers, path, precondition.map(|x| x.aff_tree))
    } else {
        afftree_from_layers_verbose(dim, layers, precondition.map(|x| x.aff_tree))
    };

    Ok(PyAffTree { aff_tree })
}

#[pyfunction]
fn read_npz(
    dim: usize,
    filename: String,
    precondition: Option<PyAffTree>,
    csv: Option<&str>,
) -> PyResult<PyAffTree> {
    let layers = match read_layers(&filename) {
        Ok(layers) => layers,
        Err(e) => {
            return Err(PyValueError::new_err(format!(
                "Error reading numpy file {}",
                e
            )))
        }
    };

    let aff_tree = if let Some(path) = csv {
        afftree_from_layers_csv(dim, &layers, path, precondition.map(|x| x.aff_tree))
    } else {
        afftree_from_layers_verbose(dim, &layers, precondition.map(|x| x.aff_tree))
    };

    Ok(PyAffTree { aff_tree })
}

#[pymodule]
fn rust_builder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let child_module = PyModule::new_bound(m.py(), "builder")?;
    child_module.add_function(wrap_pyfunction!(from_layers, m)?)?;
    child_module.add_function(wrap_pyfunction!(read_npz, m)?)?;
    m.add_submodule(&child_module);
    Ok(())
}

#[pymodule]
fn affinitree(m: &Bound<'_, PyModule>) -> PyResult<()> {
    rust_schema(m);
    rust_builder(m);
    m.add_class::<PyAffTree>()?;
    m.add_class::<AffineNodeWrapper>()?;
    m.add_class::<PyPolytope>()?;
    m.add_class::<PyAffFunc>()?;
    m.add_class::<PyArchitecture>()?;
    Ok(())
}
