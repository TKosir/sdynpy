import sys
sys.path.insert(0, './src')

import numpy as np
import sdynpy as sdpy


def _rot_z(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(4, 3)
    # Store transpose to match Geometry.global_deflection convention (rows = local axes in global)
    R[:3, :3] = np.array([[c, -s, 0],
                          [s,  c, 0],
                          [0,  0, 1]]).T
    return R


def _geometry_with_cs(rotation_deg: float) -> sdpy.Geometry:
    # Coordinate system ID 1 with rotation about Z
    cs = sdpy.coordinate_system_array(id=np.array([1]), matrix=_rot_z(rotation_deg)[np.newaxis, ...])
    # One node at origin, defined and displaced in CS 1
    nodes = sdpy.node_array(id=np.array([1]), coordinate=np.array([[0.0, 0.0, 0.0]]), def_cs=np.array([1]), disp_cs=np.array([1]))
    return sdpy.Geometry(nodes, cs)


def test_no_cross_coupling_rotations_true_missing_zeros_preserved():
    # Original: 0 deg; New: 90 deg rotation about Z
    g0 = _geometry_with_cs(0.0)
    g1 = _geometry_with_cs(90.0)

    # Shape has only translational X+ at node 1 non-zero (value 1.0); all others missing
    coords = sdpy.coordinate.from_nodelist([1], directions=[1])  # only X+
    shape = sdpy.shape_array(coordinate=coords, shape_matrix=np.array([1.0]))

    # Transform with rotations enabled and fill missing DOFs as zeros
    t_shape = shape.transform_coordinate_system(g0, g1, rotations=True, missing_dofs_are_zero=True)

    # Build full coordinate list for checking
    all_coords = sdpy.coordinate.from_nodelist([1], directions=[1, 2, 3, 4, 5, 6])
    vals = t_shape[all_coords]

    # Translations should be rotated: X -> Y (since 90deg about Z)
    # So X becomes 0, Y becomes 1, Z remains 0
    tx, ty, tz, rx, ry, rz = vals

    assert np.isclose(tx, 0.0)
    # Convention in sdynpy: +90 deg about Z maps local X to -global Y
    assert np.isclose(ty, -1.0)
    assert np.isclose(tz, 0.0)

    # Rotations should remain zero (no cross-coupling from translations)
    assert np.isclose(rx, 0.0)
    assert np.isclose(ry, 0.0)
    assert np.isclose(rz, 0.0)


def test_rotation_block_preserves_zero_when_only_rotation_present():
    # Original: 0 deg; New: 90 deg rotation about Z
    g0 = _geometry_with_cs(0.0)
    g1 = _geometry_with_cs(90.0)

    # Shape has only rotational RX+ at node 1 non-zero (value 2.0); all translations missing
    coords = sdpy.coordinate.from_nodelist([1], directions=[4])  # only RX+
    shape = sdpy.shape_array(coordinate=coords, shape_matrix=np.array([2.0]))

    t_shape = shape.transform_coordinate_system(g0, g1, rotations=True, missing_dofs_are_zero=True)

    all_coords = sdpy.coordinate.from_nodelist([1], directions=[1, 2, 3, 4, 5, 6])
    tx, ty, tz, rx, ry, rz = t_shape[all_coords]

    # Translations must remain zero
    assert np.isclose(tx, 0.0)
    assert np.isclose(ty, 0.0)
    assert np.isclose(tz, 0.0)

    # Rotations rotate like vectors: RX -> -RY at +90deg about Z
    assert np.isclose(rx, 0.0)
    assert np.isclose(ry, -2.0)
    assert np.isclose(rz, 0.0)


def _build_two_node_geometry(rot_deg_node1: float, rot_deg_node2: float) -> sdpy.Geometry:
    cs = sdpy.coordinate_system_array(id=np.array([1, 2]))
    cs.matrix[0] = _rot_z(rot_deg_node1)
    cs.matrix[1] = _rot_z(rot_deg_node2)
    nodes = sdpy.node_array(id=np.array([1, 2]),
                            coordinate=np.array([[0.0, 0.0, 0.0],
                                                 [0.0, 0.0, 0.0]]),
                            def_cs=np.array([1, 2]),
                            disp_cs=np.array([1, 2]))
    return sdpy.Geometry(nodes, cs)


def test_no_cross_coupling_across_nodes_with_mixed_dofs():
    # Node 1 rotated +90 deg about Z, Node 2 rotated -90 deg about Z
    geom = _build_two_node_geometry(90.0, -90.0)
    geom_global = _geometry_with_cs(0.0)
    # Add second node to global geometry
    geom_global.node = sdpy.node_array(id=np.array([1, 2]),
                                       coordinate=np.array([[0.0, 0.0, 0.0],
                                                            [0.0, 0.0, 0.0]]),
                                       def_cs=np.array([1, 1]),
                                       disp_cs=np.array([1, 1]))

    # Shape: translation X at node 1 = 1.0; rotation RX at node 2 = 3.0
    coords = sdpy.coordinate_array(node=np.array([1, 2]), direction=np.array([1, 4]))
    shape = sdpy.shape_array(coordinate=coords, shape_matrix=np.array([1.0, 3.0]))

    t_shape = shape.transform_coordinate_system(geom, geom_global, rotations=True, missing_dofs_are_zero=True)

    # Check outputs per node
    vals_n1 = t_shape[sdpy.coordinate.from_nodelist([1], [1, 2, 3, 4, 5, 6])]
    vals_n2 = t_shape[sdpy.coordinate.from_nodelist([2], [1, 2, 3, 4, 5, 6])]

    # Node 1 original frame is +90deg; mapping to global yields X_old -> +Y_global; no rotations at node 1
    assert np.allclose(vals_n1, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], atol=1e-12)
    # Node 2: RX (local) -> -RY (global) due to -90deg Z rotation
    assert np.allclose(vals_n2, [0.0, 0.0, 0.0, 0.0, -3.0, 0.0])


def test_convert_shape_to_global_produces_global_axes_without_mask():
    # Build geometry with 1 node rotated 90 deg, transform to global
    geom = _geometry_with_cs(90.0)
    geom_global = _geometry_with_cs(0.0)

    # Original shape only has X translation
    shape = sdpy.shape_array(coordinate=sdpy.coordinate.from_nodelist([1], [1]), shape_matrix=np.array([5.0]))
    t_shape = shape.transform_coordinate_system(geom, geom_global, rotations=True, missing_dofs_are_zero=True)

    # Should map to Y in global with same magnitude, other comps zero
    vals = t_shape[sdpy.coordinate.from_nodelist([1], [1, 2, 3, 4, 5, 6])]
    assert np.allclose(vals, [0.0, 5.0, 0.0, 0.0, 0.0, 0.0])


