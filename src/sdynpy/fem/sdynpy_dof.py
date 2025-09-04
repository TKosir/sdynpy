# -*- coding: utf-8 -*-
"""
Functions for optimally down-selecting degrees of freedom from a candidate set.

Copyright 2022 National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Iterable, Optional 
import numpy as np
from sdynpy.core.sdynpy_geometry import (
    node_array,        # builds NodeArray
    element_array,     # builds ElementArray
    NodeArray,
    ElementArray,
)

UPDATE_TIME = 5


def by_condition_number(sensors_to_keep, shape_matrix, return_condition_numbers=False):
    '''Get the best set of degrees of freedom by mode shape condition number

    This function accepts a shape matrix and returns the set of degrees of
    freedom that corresponds to the lowest condition number.

    Parameters
    ----------
    sensors_to_keep : int
        The number of sensors to keep.
    shape_matrix : np.ndarray
        A 2D or 3D numpy array.  If it is a 2D array, the row indices should
        correspond to the degree of freedom and the column indices should
        correspond to the mode.  For a 3D array, the first index corresponds to
        a "bundle" of channels (e.g. a triaxial accelerometer) that must be
        kept or removed as one unit.
    return_condition_numbers : bool (default False)
        If True, return a second value that is the condition number at each
        iteration of the technique.

    Returns
    -------
    indices : np.array
        A 1d array corresponding to the indices to keep in the first dimension
        of the shape_matrix array (e.g. new_shape_matrix =
        shape_matrix[incies,...])
    returned_condition_numbers : list
        The condition number at each iteration.  Returned only if
        return_condition_numbers is True.
    '''
    shape_matrix = shape_matrix.copy()
    keep_indices = np.arange(shape_matrix.shape[0])
    if return_condition_numbers:
        returned_condition_numbers = [np.linalg.cond(
            shape_matrix.reshape(-1, shape_matrix.shape[-1]))]
    start_time = time.time()
    while shape_matrix.shape[0] > sensors_to_keep:
        condition_numbers = [np.linalg.cond(shape_matrix[np.arange(shape_matrix.shape[0]) != removed_dof_index, ...].reshape(
            -1, shape_matrix.shape[-1])) for removed_dof_index in range(shape_matrix.shape[0])]
        dof_to_remove = np.argmin(condition_numbers)
#        print('Condition Numbers {:}'.format(condition_numbers))
#        print('Removing DOF {:}'.format(dof_to_remove))
        shape_matrix = np.delete(shape_matrix.copy(), dof_to_remove, axis=0)
        keep_indices = np.delete(keep_indices, dof_to_remove, axis=0)
        if return_condition_numbers:
            returned_condition_numbers.append(condition_numbers[dof_to_remove])
        new_time = time.time()
        if new_time - start_time > UPDATE_TIME:
            print('{:} DoFs Remaining'.format(shape_matrix.shape[0]))
            start_time += UPDATE_TIME
    if return_condition_numbers:
        return keep_indices, returned_condition_numbers
    else:
        return keep_indices


def by_effective_independence(sensors_to_keep, shape_matrix, return_efi=False):
    '''Get the best set of degrees of freedom by mode shape effective independence

    This function accepts a shape matrix and returns the set of degrees of
    freedom that corresponds to the maximum effective independence.

    Parameters
    ----------
    sensors_to_keep : int
        The number of sensors to keep.
    shape_matrix : np.ndarray
        A 2D or 3D numpy array.  If it is a 2D array, the row indices should
        correspond to the degree of freedom and the column indices should
        correspond to the mode.  For a 3D array, the first index corresponds to
        a "bundle" of channels (e.g. a triaxial accelerometer) that must be
        kept or removed as one unit.
    return_efi : bool (default False)
        If True, return a second value that is the effective independence at
        each iteration of the technique.

    Returns
    -------
    indices : np.array
        A 1d array corresponding to the indices to keep in the first dimension
        of the shape_matrix array (e.g. new_shape_matrix =
        shape_matrix[incies,...])
    returned_efi : list
        The effective independence at each iteration.  Returned only if
        return_efi is set to True.
    '''
    shape_matrix = shape_matrix.copy()
    keep_indices = np.arange(shape_matrix.shape[0])
    if return_efi:
        returned_efi = []
    start_time = time.time()
    while shape_matrix.shape[0] > sensors_to_keep:
        Q = shape_matrix.reshape(-1, shape_matrix.shape[-1]
                                 ).T @ shape_matrix.reshape(-1, shape_matrix.shape[-1])
        if return_efi:
            returned_efi.append(np.linalg.det(Q))
        Qinv = np.linalg.inv(Q)
        if shape_matrix.ndim == 2:
            EfIs = np.diag(shape_matrix @ Qinv @ shape_matrix.T)
        else:
            EfIs = 1 - np.linalg.det(np.eye(3) - np.einsum('ijk,kl,iml->ijm',
                                     shape_matrix, Qinv, shape_matrix))
        dof_to_remove = np.argmin(EfIs)
#        print('Effective Independences {:}'.format(EfI3s))
#        print('Removing DOF {:}'.format(dof_to_remove))
        shape_matrix = np.delete(shape_matrix.copy(), dof_to_remove, axis=0)
        keep_indices = np.delete(keep_indices, dof_to_remove, axis=0)
        new_time = time.time()
        if new_time - start_time > UPDATE_TIME:
            print('{:} DoFs Remaining'.format(shape_matrix.shape[0]))
            start_time += UPDATE_TIME
    if return_efi:
        return keep_indices, returned_efi
    else:
        return keep_indices


class BoundaryNodeExtractor:
    # ---- Face templates (corner nodes only) ----
    _TET_CORNER_FACE_IDS: List[List[int]] = [
        [0, 2, 1],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
    ]
    _WEDGE_CORNER_FACE_IDS: List[List[int]] = [
        [0, 2, 1],       # tri
        [3, 4, 5],       # tri
        [0, 1, 4, 3],    # quad
        [1, 2, 5, 4],    # quad
        [2, 0, 3, 5],    # quad
    ]
    _HEX_CORNER_FACE_IDS: List[List[int]] = [
        [0, 3, 2, 1],    # bottom
        [4, 5, 6, 7],    # top
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ]

    # ---- SDynPy type codes ----
    # Solid tetrahedron types (linear and quadratic)
    _SOLID_TET_TYPES   = {111, 118}
    # Solid wedge types (linear, parabolic, cubic)
    _SOLID_WEDGE_TYPES = {112, 113, 114}
    # Solid hexahedron types (linear, parabolic, cubic)
    _SOLID_HEX_TYPES   = {115, 116, 117}
    
    # Thick shell elements (3D elements acting as shells)
    _THICK_SHELL_WEDGE_TYPES = {101, 102, 103}  # Linear, Parabolic, Cubic Wedge
    _THICK_SHELL_HEX_TYPES   = {104, 105, 106}  # Linear, Parabolic, Cubic Brick
    
    # Shell elements (2D - already boundary, just pass through)
    _THIN_SHELL_TRI_TYPES  = {91, 92, 93}   # Linear, Parabolic, Cubic Triangle
    _THIN_SHELL_QUAD_TYPES = {94, 95, 96}   # Linear, Parabolic, Cubic Quadrilateral
    _PLATE_TRI_TYPES       = {61, 62, 63}   # Linear, Parabolic, Cubic Triangle
    _PLATE_QUAD_TYPES      = {64, 65, 66}   # Linear, Parabolic, Cubic Quadrilateral
    _MEMBRANE_TRI_TYPES    = {72, 73, 74}   # Parabolic, Cubic, Linear Triangle
    _MEMBRANE_QUAD_TYPES   = {71, 75, 76}   # Linear, Parabolic, Cubic Quadrilateral
    _PLANE_STRESS_TRI_TYPES  = {41, 42, 43}   # Linear, Parabolic, Cubic Triangle
    _PLANE_STRESS_QUAD_TYPES = {44, 45, 46}   # Linear, Parabolic, Cubic Quadrilateral
    _PLANE_STRAIN_TRI_TYPES  = {51, 52, 53}   # Linear, Parabolic, Cubic Triangle
    _PLANE_STRAIN_QUAD_TYPES = {54, 55, 56}   # Linear, Parabolic, Cubic Quadrilateral
    _AXISYM_SOLID_TRI_TYPES  = {81, 82}       # Linear, Parabolic Triangle
    _AXISYM_SOLID_QUAD_TYPES = {84, 85}       # Linear, Parabolic Quadrilateral
    _AXISYM_SHELL_TYPES      = {171, 172}     # Linear, Parabolic Shell

    # Output 2D element types for plotting
    _PLANE_STRESS_TRI  = 41   # 3-node triangle
    _PLANE_STRESS_QUAD = 44   # 4-node quad

    """
    Comprehensive boundary node extractor supporting all SDynPy element types.

    Behavior:
    - For 3D elements (solid tetrahedra, wedges, hexahedra, thick shells):
      Extracts boundary faces using only corner nodes (linearized).
    - For 2D elements (thin shells, plates, membranes, plane stress/strain):
      Passes through the elements as they are already boundaries.
    - Supports both linear and higher-order elements by using only corner nodes.
    - Unknown element types raise an error.
    
    Filtering:
    - If boundary_node_ids is provided, restricts extraction to the subset of elements and
      faces whose CORNER nodes are all within that set; boundary-ness is computed only
      within this restricted subset (faces shared by two subset elements are removed).
      Midside IDs in boundary_node_ids are ignored implicitly by using corner nodes only.
    - include_node_ids are validated against the input NodeArray IDs. After boundary
      extraction, all include_node_ids are ensured to be present in the output NodeArray,
      without adding any connectivity. If an include node is already part of kept faces,
      it is already included and left as-is.

    Supported Elements:
    - Solid: Tetrahedra (111, 118), Wedges (112, 113, 114), Hexahedra (115, 116, 117)
    - Thick Shell: Wedges (101, 102, 103), Bricks (104, 105, 106)
    - Thin Shell: Triangles (91, 92, 93), Quads (94, 95, 96)
    - Plate: Triangles (61, 62, 63), Quads (64, 65, 66)
    - Membrane: Triangles (72, 73, 74), Quads (71, 75, 76)
    - Plane Stress: Triangles (41, 42, 43), Quads (44, 45, 46)
    - Plane Strain: Triangles (51, 52, 53), Quads (54, 55, 56)
    - Axisymmetric: Triangles (81, 82), Quads (84, 85), Shells (171, 172)

    API:
      BoundaryNodeExtractor(node_in, elem_in, boundary_node_ids=None, include_node_ids=None)
      .get_node_out() -> NodeArray
      .get_elem_out() -> ElementArray
    """

    def _validate_node_array(self, node_in: NodeArray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Validate and extract fields from a SDynPy NodeArray.
        
        Parameters
        ----------
        node_in : NodeArray
            Input node array to validate and extract fields from.
            
        Returns
        -------
        tuple
            (node_ids, node_xyz, node_color, node_def_cs, node_disp_cs)
            All required and optional fields from the node array.
            
        Raises
        ------
        TypeError
            If node_in is not a structured numpy array.
        ValueError
            If required fields ('id', 'coordinate') are missing.
        """
        if not isinstance(node_in, np.ndarray) or node_in.dtype.names is None:
            raise TypeError("node_in must be a SDynPy NodeArray (structured array).")
        node_ids  = self._safe_field(node_in, 'id')
        node_xyz  = self._safe_field(node_in, 'coordinate')
        if node_ids is None or node_xyz is None:
            raise ValueError("NodeArray must have 'id' and 'coordinate' fields.")
        node_color  = self._safe_field(node_in, 'color')
        node_def_cs = self._safe_field(node_in, 'def_cs')
        node_disp_cs= self._safe_field(node_in, 'disp_cs')
        return node_ids, node_xyz, node_color, node_def_cs, node_disp_cs

    def _validate_element_array(self, elem_in: ElementArray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and extract fields from a SDynPy ElementArray.
        
        Parameters
        ----------
        elem_in : ElementArray
            Input element array to validate and extract fields from.
            
        Returns
        -------
        tuple
            (elem_type, elem_conn) - Element types and connectivity arrays.
            
        Raises
        ------
        TypeError
            If elem_in is not a structured numpy array.
        ValueError
            If required fields ('type', 'connectivity') are missing.
        """
        if not isinstance(elem_in, np.ndarray) or elem_in.dtype.names is None:
            raise TypeError("elem_in must be a SDynPy ElementArray (structured array).")
        elem_type = self._safe_field(elem_in, 'type')
        elem_conn = self._safe_field(elem_in, 'connectivity')
        if elem_type is None or elem_conn is None:
            raise ValueError("ElementArray must have 'type' and 'connectivity' fields.")
        return elem_type, elem_conn

    def _build_node_array_from_ids(
        self,
        used_ids: np.ndarray,
        id2idx: Dict[int, int],
        node_xyz: np.ndarray,
        node_color: Optional[np.ndarray],
        node_def_cs: Optional[np.ndarray],
        node_disp_cs: Optional[np.ndarray],
    ) -> NodeArray:
        """
        Build a NodeArray from selected node IDs and original node data.
        
        Parameters
        ----------
        used_ids : np.ndarray
            Array of node IDs to include in the output.
        id2idx : Dict[int, int]
            Mapping from node ID to index in original arrays.
        node_xyz : np.ndarray
            Original node coordinates array.
        node_color : Optional[np.ndarray]
            Original node colors array (may be None).
        node_def_cs : Optional[np.ndarray]
            Original node definition coordinate system array (may be None).
        node_disp_cs : Optional[np.ndarray]
            Original node displacement coordinate system array (may be None).
            
        Returns
        -------
        NodeArray
            New NodeArray containing only the selected nodes with all available fields.
        """
        used_idx = np.array([id2idx[int(n)] for n in used_ids], dtype=np.int64)
        kwargs = dict(id=used_ids, coordinate=node_xyz[used_idx])
        if node_color is not None:
            kwargs["color"] = node_color[used_idx]
        if node_def_cs is not None:
            kwargs["def_cs"] = node_def_cs[used_idx]
        if node_disp_cs is not None:
            kwargs["disp_cs"] = node_disp_cs[used_idx]
        return node_array(**kwargs)

    def _iterate_build_face_map(
        self,
        elem_type: np.ndarray,
        elem_conn: np.ndarray,
        boundary_set: Optional[set[int]],
    ) -> Dict[Tuple[int, ...], List[int]]:
        """
        Build a map of faces to corner node IDs for boundary extraction.
        
        This is the core method that processes all elements and extracts boundary faces.
        Faces that are shared between elements are marked as None and removed.
        
        Parameters
        ----------
        elem_type : np.ndarray
            Array of element type codes.
        elem_conn : np.ndarray
            Array of element connectivity arrays.
        boundary_set : Optional[set[int]]
            Set of node IDs to restrict boundary extraction to. If provided:
            1. Only elements with at least one corner node in this set are considered
            2. Only faces with all corner nodes in this set are kept in final output
            If None, process all elements normally.
            
        Returns
        -------
        Dict[Tuple[int, ...], List[int]]
            Mapping from sorted node ID tuples to original face node IDs.
            Only contains boundary faces (not shared between elements).
            
        Algorithm
        ---------
        For 3D elements (solid, thick shell):
            - Extract all faces using corner nodes only
            - Faces shared between elements are removed
            
        For 2D elements (thin shell, plate, membrane):
            - Elements are passed through as they are already boundaries
            - No face extraction needed
            
        Raises
        ------
        ValueError
            If an unsupported element type is encountered.
        """
        face_map: Dict[Tuple[int, ...], Optional[List[int]]] = {}

        for et, conn in zip(elem_type, elem_conn):
            et = int(et)
            
            # 3D Solid Elements - Extract boundary faces
            if et in self._SOLID_TET_TYPES or et in self._THICK_SHELL_WEDGE_TYPES:
                if len(conn) < 4:
                    continue
                ncorners = 4 if et in self._SOLID_TET_TYPES else 6
                corner = list(map(int, conn[:ncorners]))
                # Skip element if no corner nodes are in boundary_set
                if boundary_set is not None and not any((c in boundary_set) for c in corner):
                    continue
                
                if et in self._SOLID_TET_TYPES:
                    face_templates = self._TET_CORNER_FACE_IDS
                else:  # thick shell wedge
                    face_templates = self._WEDGE_CORNER_FACE_IDS
                    
                for loc in face_templates:
                    face_ids = [corner[i] for i in loc]
                    key = tuple(sorted(face_ids))
                    if key in face_map:
                        face_map[key] = None
                    else:
                        face_map[key] = face_ids

            elif et in self._SOLID_WEDGE_TYPES:
                if len(conn) < 6:
                    continue
                ncorners = 6
                corner = list(map(int, conn[:ncorners]))
                # Skip element if no corner nodes are in boundary_set
                if boundary_set is not None and not any((c in boundary_set) for c in corner):
                    continue
                for loc in self._WEDGE_CORNER_FACE_IDS:
                    face_ids = [corner[i] for i in loc]
                    key = tuple(sorted(face_ids))
                    if key in face_map:
                        face_map[key] = None
                    else:
                        face_map[key] = face_ids

            elif et in self._SOLID_HEX_TYPES or et in self._THICK_SHELL_HEX_TYPES:
                if len(conn) < 8:
                    continue
                ncorners = 8
                corner = list(map(int, conn[:ncorners]))
                # Skip element if no corner nodes are in boundary_set
                if boundary_set is not None and not any((c in boundary_set) for c in corner):
                    continue
                for loc in self._HEX_CORNER_FACE_IDS:
                    face_ids = [corner[i] for i in loc]
                    key = tuple(sorted(face_ids))
                    if key in face_map:
                        face_map[key] = None
                    else:
                        face_map[key] = face_ids

            # 2D Shell/Plate/Membrane Elements - Pass through as boundary
            elif (et in self._THIN_SHELL_TRI_TYPES or et in self._PLATE_TRI_TYPES or 
                  et in self._MEMBRANE_TRI_TYPES or et in self._PLANE_STRESS_TRI_TYPES or
                  et in self._PLANE_STRAIN_TRI_TYPES or et in self._AXISYM_SOLID_TRI_TYPES):
                # Triangle elements - use first 3 corner nodes
                if len(conn) < 3:
                    continue
                corner = list(map(int, conn[:3]))
                if boundary_set is not None and not any((c in boundary_set) for c in corner):
                    continue
                # For 2D elements, the element itself is the "face"
                key = tuple(sorted(corner))
                if key not in face_map:  # Don't mark as None (shared) since these are already boundary
                    face_map[key] = corner

            elif (et in self._THIN_SHELL_QUAD_TYPES or et in self._PLATE_QUAD_TYPES or 
                  et in self._MEMBRANE_QUAD_TYPES or et in self._PLANE_STRESS_QUAD_TYPES or
                  et in self._PLANE_STRAIN_QUAD_TYPES or et in self._AXISYM_SOLID_QUAD_TYPES or
                  et in self._AXISYM_SHELL_TYPES):
                # Quadrilateral elements - use first 4 corner nodes
                if len(conn) < 4:
                    continue
                corner = list(map(int, conn[:4]))
                if boundary_set is not None and not any((c in boundary_set) for c in corner):
                    continue
                # For 2D elements, the element itself is the "face"
                key = tuple(sorted(corner))
                if key not in face_map:  # Don't mark as None (shared) since these are already boundary
                    face_map[key] = corner

            else:
                raise ValueError(f"Unsupported element type encountered: {et}")

        # Filter final faces: keep only boundary faces with all nodes in boundary_set
        kept: Dict[Tuple[int, ...], List[int]] = {}
        for k, v in face_map.items():
            if v is not None:
                if boundary_set is None or all((nid in boundary_set) for nid in v):
                    kept[k] = v
        return kept

    def _build_elements_from_faces(self, kept_faces: List[List[int]]) -> ElementArray:
        """
        Build an ElementArray from a list of boundary faces.
        
        Parameters
        ----------
        kept_faces : List[List[int]]
            List of faces, where each face is a list of node IDs.
            Faces can be triangular (3 nodes) or quadrilateral (4 nodes).
            
        Returns
        -------
        ElementArray
            New ElementArray containing the boundary faces as 2D elements.
            Triangular faces become type 41 (plane stress triangles).
            Quadrilateral faces become type 44 (plane stress quads).
            
        Raises
        ------
        ValueError
            If a face has an unsupported number of nodes (not 3 or 4).
            
        Notes
        -----
        Element IDs are assigned sequentially starting from 1.
        All elements are assigned color 1.
        Triangular elements are placed before quadrilateral elements.
        """
        tri_conns: List[Tuple[int, int, int]] = []
        quad_conns: List[Tuple[int, int, int, int]] = []
        for face_ids in kept_faces:
            if len(face_ids) == 3:
                tri_conns.append((int(face_ids[0]), int(face_ids[1]), int(face_ids[2])))
            elif len(face_ids) == 4:
                quad_conns.append((int(face_ids[0]), int(face_ids[1]), int(face_ids[2]), int(face_ids[3])))
            else:
                raise ValueError("Face with unsupported number of nodes encountered.")

        n_tri, n_quad = len(tri_conns), len(quad_conns)
        n_tot = n_tri + n_quad
        if n_tot == 0:
            return element_array(
                id=np.array([], dtype=np.uint64),
                type=np.array([], dtype=np.uint8),
                color=np.array([], dtype=np.uint16),
                connectivity=np.array([], dtype=object),
            )

        elem_ids_out   = np.arange(1, n_tot + 1, dtype=np.uint64)
        elem_types_out = np.empty(n_tot, dtype=np.uint8)
        if n_tri:
            elem_types_out[:n_tri] = self._PLANE_STRESS_TRI
        if n_quad:
            elem_types_out[n_tri:] = self._PLANE_STRESS_QUAD
        elem_colors_out = np.ones(n_tot, dtype=np.uint16)

        connectivity_out = np.empty(n_tot, dtype=object)
        for i, c in enumerate(tri_conns + quad_conns):
            connectivity_out[i] = np.array(c, dtype=np.uint64)

        return element_array(
            id=elem_ids_out,
            type=elem_types_out,
            color=elem_colors_out,
            connectivity=connectivity_out,
        )

    def __init__(
        self,
        node_in: NodeArray,
        elem_in: ElementArray,
        boundary_node_ids: Optional[Iterable[int]] = None,
        include_node_ids: Optional[Iterable[int]] = None,
    ) -> None:
        """
        Initialize the boundary node extractor and perform extraction.
        
        Parameters
        ----------
        node_in : NodeArray
            Input nodes containing geometry and properties.
        elem_in : ElementArray
            Input elements defining connectivity and types.
        boundary_node_ids : Optional[Iterable[int]], default=None
            Node IDs to restrict boundary extraction to. If provided:
            - Only elements with corner nodes in this set are considered
            - Only faces with all corner nodes in this set are kept
            - Effectively allows extraction of boundaries within a subset
        include_node_ids : Optional[Iterable[int]], default=None
            Additional node IDs to include in output without connectivity.
            Useful for including isolated nodes or reference points.
            
        Raises
        ------
        TypeError
            If node_in or elem_in are not proper SDynPy arrays.
        ValueError
            If required fields are missing, node IDs are not unique,
            or include_node_ids reference non-existent nodes.
            Also raised if unsupported element types are encountered.
            
        Notes
        -----
        The extraction is performed immediately during initialization.
        Results are stored and accessed via get_node_out() and get_elem_out().
        
        For 3D elements: Boundary faces are extracted using corner nodes only.
        For 2D elements: Elements are passed through as they're already boundaries.
        Higher-order elements are linearized by using only corner nodes.
        """
        # Validate inputs
        node_ids, node_xyz, node_color, node_def_cs, node_disp_cs = self._validate_node_array(node_in)
        elem_type, elem_conn = self._validate_element_array(elem_in)

        # Ensure unique node ids and index map
        uniq, counts = np.unique(node_ids, return_counts=True)
        if np.any(counts != 1):
            raise ValueError("NodeArray: 'id' values must be unique.")
        id2idx: Dict[int, int] = {int(i): k for k, i in enumerate(node_ids)}

        # Prepare boundary subset (hard filter using CORNER node IDs only). Ignore midside implicitly.
        boundary_set: Optional[set[int]] = None
        if boundary_node_ids is not None:
            if not self._is_sequence_of_ints(boundary_node_ids):
                raise ValueError("boundary_node_ids must be an iterable of integers (node IDs).")
            boundary_set = {int(v) for v in boundary_node_ids if int(v) in id2idx}

        # Validate include nodes exist upfront
        include_set: Optional[set[int]] = None
        if include_node_ids is not None:
            if not self._is_sequence_of_ints(include_node_ids):
                raise ValueError("include_node_ids must be an iterable of integers (node IDs).")
            include_set = set(int(v) for v in include_node_ids)
            missing = [int(v) for v in include_set if int(v) not in id2idx]
            if len(missing) > 0:
                raise ValueError(f"include_node_ids contain IDs not present in the input NodeArray: {missing[:5]}{'...' if len(missing) > 5 else ''}")

        # Build face map within subset and collect kept faces
        face_map = self._iterate_build_face_map(elem_type, elem_conn, boundary_set)
        kept_faces_list = list(face_map.values())

        # Aggregate used node IDs from faces
        used_id_set: set[int] = set()
        for f in kept_faces_list:
            for nid in f:
                used_id_set.add(int(nid))

        # Add include nodes (without connectivity) after extraction
        if include_set is not None and len(include_set) > 0:
            used_id_set.update(int(n) for n in include_set)

        # Build NodeArray
        used_ids = np.array(sorted(used_id_set), dtype=np.uint64)
        node_out = self._build_node_array_from_ids(used_ids, id2idx, node_xyz, node_color, node_def_cs, node_disp_cs)

        # Build ElementArray
        elem_out = self._build_elements_from_faces(kept_faces_list)

        self.node_out = node_out
        self.elem_out = elem_out

    def get_node_out(self) -> NodeArray:
        """
        Get the extracted boundary nodes.
        
        Returns
        -------
        NodeArray
            Boundary nodes with all available fields (coordinates, colors, etc.).
            Includes nodes from boundary faces plus any requested include_node_ids.
        """
        return self.node_out
    
    def get_elem_out(self) -> ElementArray:
        """
        Get the extracted boundary elements.
        
        Returns
        -------
        ElementArray
            Boundary elements as 2D triangles (type 41) and quads (type 44).
            For 3D elements: boundary faces extracted from solid elements.
            For 2D elements: original elements passed through.
        """
        return self.elem_out
    
    @staticmethod
    def _is_sequence_of_ints(x: Iterable[int]) -> bool:
        """
        Check if an iterable contains only values convertible to integers.
        
        Parameters
        ----------
        x : Iterable[int]
            Iterable to check for integer-convertible values.
            
        Returns
        -------
        bool
            True if all values can be converted to integers, False otherwise.
        """
        try:
            for v in x:
                _ = int(v)
            return True
        except Exception:
            return False

    @staticmethod
    def _safe_field(a: np.ndarray, name: str, default=None):
        """
        Safely extract a field from a structured numpy array.
        
        Parameters
        ----------
        a : np.ndarray
            Structured numpy array to extract field from.
        name : str
            Name of the field to extract.
        default : Any, default=None
            Value to return if field doesn't exist.
            
        Returns
        -------
        Any
            Field data if it exists, otherwise default value.
        """
        return a[name] if (a.dtype.names and name in a.dtype.names) else default

