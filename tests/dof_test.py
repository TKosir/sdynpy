import pytest
import numpy as np
from sdynpy.core.sdynpy_geometry import node_array, element_array
from sdynpy.fem.sdynpy_dof import BoundaryNodeExtractor


class TestBoundaryNodeExtractor:
    def create_simple_tet_mesh(self):
        # Single tetrahedron with 4 nodes
        node_ids = np.array([1, 2, 3, 4], dtype=np.uint64)
        coordinates = np.array([
            [0.0, 0.0, 0.0],  # node 1
            [1.0, 0.0, 0.0],  # node 2
            [0.0, 1.0, 0.0],  # node 3
            [0.0, 0.0, 1.0],  # node 4
        ], dtype=np.float64)

        nodes = node_array(id=node_ids, coordinate=coordinates)

        # Single tet element (type 111)
        elem_ids = np.array([1], dtype=np.uint64)
        elem_types = np.array([111], dtype=np.uint8)
        elem_colors = np.array([1], dtype=np.uint16)
        connectivity = np.array([np.array([1, 2, 3, 4], dtype=np.uint64)], dtype=object)

        elements = element_array(
            id=elem_ids,
            type=elem_types,
            color=elem_colors,
            connectivity=connectivity
        )

        return nodes, elements

    def create_two_tet_mesh(self):
        # Two tetrahedra sharing face [1,2,3]
        node_ids = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
        coordinates = np.array([
            [0.0, 0.0, 0.0],   # 1
            [1.0, 0.0, 0.0],   # 2
            [0.0, 1.0, 0.0],   # 3
            [0.0, 0.0, 1.0],   # 4
            [0.0, 0.0, -1.0],  # 5
        ], dtype=np.float64)

        nodes = node_array(id=node_ids, coordinate=coordinates)

        elem_ids = np.array([1, 2], dtype=np.uint64)
        elem_types = np.array([111, 111], dtype=np.uint8)
        elem_colors = np.array([1, 1], dtype=np.uint16)
        connectivity = np.array([
            np.array([1, 2, 3, 4], dtype=np.uint64),  # first tet
            np.array([1, 3, 2, 5], dtype=np.uint64),  # second tet (reversed winding on shared face)
        ], dtype=object)

        elements = element_array(
            id=elem_ids,
            type=elem_types,
            color=elem_colors,
            connectivity=connectivity
        )

        return nodes, elements

    def create_hex_mesh(self):
        # Single hexahedron with 8 nodes (type 115)
        node_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint64)
        coordinates = np.array([
            [0.0, 0.0, 0.0],  # 1
            [1.0, 0.0, 0.0],  # 2
            [1.0, 1.0, 0.0],  # 3
            [0.0, 1.0, 0.0],  # 4
            [0.0, 0.0, 1.0],  # 5
            [1.0, 0.0, 1.0],  # 6
            [1.0, 1.0, 1.0],  # 7
            [0.0, 1.0, 1.0],  # 8
        ], dtype=np.float64)

        nodes = node_array(id=node_ids, coordinate=coordinates)

        elem_ids = np.array([1], dtype=np.uint64)
        elem_types = np.array([115], dtype=np.uint8)
        elem_colors = np.array([1], dtype=np.uint16)
        connectivity = np.array([np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint64)], dtype=object)

        elements = element_array(
            id=elem_ids,
            type=elem_types,
            color=elem_colors,
            connectivity=connectivity
        )

        return nodes, elements

    def test_single_tet_extraction(self):
        nodes, elements = self.create_simple_tet_mesh()
        extractor = BoundaryNodeExtractor(nodes, elements)
        node_out = extractor.get_node_out()
        elem_out = extractor.get_elem_out()

        assert len(node_out) == 4
        assert set(node_out['id']) == {1, 2, 3, 4}

        assert len(elem_out) == 4
        assert all(elem_out['type'] == 41)  # triangles
        for conn in elem_out['connectivity']:
            assert len(conn) == 3

    def test_two_tet_shared_face(self):
        nodes, elements = self.create_two_tet_mesh()
        extractor = BoundaryNodeExtractor(nodes, elements)
        node_out = extractor.get_node_out()
        elem_out = extractor.get_elem_out()

        assert len(node_out) == 5
        assert set(node_out['id']) == {1, 2, 3, 4, 5}

        assert len(elem_out) == 6
        assert all(elem_out['type'] == 41)

    def test_hex_extraction(self):
        nodes, elements = self.create_hex_mesh()
        extractor = BoundaryNodeExtractor(nodes, elements)
        node_out = extractor.get_node_out()
        elem_out = extractor.get_elem_out()

        assert len(node_out) == 8
        assert set(node_out['id']) == {1, 2, 3, 4, 5, 6, 7, 8}

        assert len(elem_out) == 6
        assert all(elem_out['type'] == 44)
        for conn in elem_out['connectivity']:
            assert len(conn) == 4

    def test_boundary_node_filtering(self):
        nodes, elements = self.create_two_tet_mesh()
        boundary_ids = [1, 2, 3]
        extractor = BoundaryNodeExtractor(nodes, elements, boundary_node_ids=boundary_ids)
        node_out = extractor.get_node_out()
        elem_out = extractor.get_elem_out()

        node_ids_in_faces = set()
        for conn in elem_out['connectivity']:
            node_ids_in_faces.update(conn)
        assert node_ids_in_faces.issubset({1, 2, 3})

        if len(elem_out) > 0:
            assert {1, 2, 3}.issubset(set(node_out['id']))

        # Shared face [1,2,3] is interior in the subset, so no faces remain
        assert len(elem_out) == 0
        assert len(node_out) == 0

    def test_boundary_node_filtering_with_results(self):
        nodes, elements = self.create_simple_tet_mesh()
        boundary_ids = [1, 2, 3]
        extractor = BoundaryNodeExtractor(nodes, elements, boundary_node_ids=boundary_ids)
        node_out = extractor.get_node_out()
        elem_out = extractor.get_elem_out()

        assert len(elem_out) == 1
        assert elem_out['type'][0] == 41
        assert set(node_out['id']) == {1, 2, 3}
        assert set(elem_out['connectivity'][0]) == {1, 2, 3}

    def test_include_node_ids(self):
        nodes, elements = self.create_simple_tet_mesh()

        extra_node_ids = np.array([1, 2, 3, 4, 10], dtype=np.uint64)
        extra_coords = np.vstack([nodes['coordinate'], [[5.0, 5.0, 5.0]]])
        nodes_with_extra = node_array(id=extra_node_ids, coordinate=extra_coords)

        include_ids = [10]
        extractor = BoundaryNodeExtractor(nodes_with_extra, elements, include_node_ids=include_ids)
        node_out = extractor.get_node_out()
        elem_out = extractor.get_elem_out()

        assert len(node_out) == 5
        assert 10 in node_out['id']
        assert len(elem_out) == 4

    def test_include_node_not_in_input_error(self):
        nodes, elements = self.create_simple_tet_mesh()
        with pytest.raises(ValueError, match="include_node_ids contain IDs not present"):
            BoundaryNodeExtractor(nodes, elements, include_node_ids=[999])

    def test_unsupported_element_type_error(self):
        nodes, _ = self.create_simple_tet_mesh()

        elem_ids = np.array([1], dtype=np.uint64)
        elem_types = np.array([99], dtype=np.uint8)  # unsupported
        elem_colors = np.array([1], dtype=np.uint16)
        connectivity = np.array([np.array([1, 2, 3], dtype=np.uint64)], dtype=object)

        elements = element_array(
            id=elem_ids,
            type=elem_types,
            color=elem_colors,
            connectivity=connectivity
        )

        with pytest.raises(ValueError, match="Unsupported element type encountered"):
            BoundaryNodeExtractor(nodes, elements)

    def test_empty_boundary_set(self):
        nodes, elements = self.create_simple_tet_mesh()

        # add isolated node 10 to nodes only
        extra_node_ids = np.array([1, 2, 3, 4, 10], dtype=np.uint64)
        extra_coords = np.vstack([nodes['coordinate'], [[5.0, 5.0, 5.0]]])
        nodes_with_extra = node_array(id=extra_node_ids, coordinate=extra_coords)

        extractor = BoundaryNodeExtractor(nodes_with_extra, elements, boundary_node_ids=[10])
        node_out = extractor.get_node_out()
        elem_out = extractor.get_elem_out()

        assert len(elem_out) == 0
        assert len(node_out) == 0

    def test_empty_mesh(self):
        empty_nodes = node_array(
            id=np.array([], dtype=np.uint64),
            coordinate=np.zeros((0, 3), dtype=np.float64)
        )
        empty_elements = element_array(
            id=np.array([], dtype=np.uint64),
            type=np.array([], dtype=np.uint8),
            color=np.array([], dtype=np.uint16),
            connectivity=np.array([], dtype=object)
        )

        extractor = BoundaryNodeExtractor(empty_nodes, empty_elements)
        node_out = extractor.get_node_out()
        elem_out = extractor.get_elem_out()

        assert len(node_out) == 0
        assert len(elem_out) == 0

    def test_node_metadata_preservation(self):
        nodes, elements = self.create_simple_tet_mesh()

        node_colors = np.array([1, 2, 3, 4], dtype=np.uint16)
        nodes_with_color = node_array(
            id=nodes['id'],
            coordinate=nodes['coordinate'],
            color=node_colors
        )

        extractor = BoundaryNodeExtractor(nodes_with_color, elements)
        node_out = extractor.get_node_out()

        assert 'color' in node_out.dtype.names
        assert list(node_out['color']) == [1, 2, 3, 4]

    def test_duplicate_node_ids_error(self):
        node_ids = np.array([1, 2, 2, 4], dtype=np.uint64)
        coordinates = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],  # duplicate id 2
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        nodes = node_array(id=node_ids, coordinate=coordinates)
        _, elements = self.create_simple_tet_mesh()

        with pytest.raises(ValueError, match="'id' values must be unique"):
            BoundaryNodeExtractor(nodes, elements)

    def test_boundary_and_include_combined(self):
        nodes, elements = self.create_two_tet_mesh()

        extra_node_ids = np.array([1, 2, 3, 4, 5, 10], dtype=np.uint64)
        extra_coords = np.vstack([nodes['coordinate'], [[5.0, 5.0, 5.0]]])
        nodes_with_extra = node_array(id=extra_node_ids, coordinate=extra_coords)

        boundary_ids = [1, 2, 3, 4]
        include_ids = [10]

        extractor = BoundaryNodeExtractor(
            nodes_with_extra, elements,
            boundary_node_ids=boundary_ids,
            include_node_ids=include_ids
        )
        node_out = extractor.get_node_out()
        elem_out = extractor.get_elem_out()

        assert 10 in node_out['id']

        face_nodes = set()
        for conn in elem_out['connectivity']:
            face_nodes.update(conn)
        assert face_nodes.issubset({1, 2, 3, 4})

    # ---- New tests for additional element types ----

    def create_quadratic_tet_mesh(self):
        """Create a quadratic tetrahedron mesh (10 nodes)."""
        # 10 nodes: 4 corners + 6 midside
        node_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint64)
        coordinates = np.array([
            [0.0, 0.0, 0.0],  # 1 - corner
            [1.0, 0.0, 0.0],  # 2 - corner
            [0.0, 1.0, 0.0],  # 3 - corner
            [0.0, 0.0, 1.0],  # 4 - corner
            [0.5, 0.0, 0.0],  # 5 - midside 1-2
            [0.5, 0.5, 0.0],  # 6 - midside 2-3
            [0.0, 0.5, 0.0],  # 7 - midside 3-1
            [0.0, 0.0, 0.5],  # 8 - midside 1-4
            [0.5, 0.0, 0.5],  # 9 - midside 2-4
            [0.0, 0.5, 0.5],  # 10 - midside 3-4
        ], dtype=np.float64)
        
        nodes = node_array(id=node_ids, coordinate=coordinates)
        
        # Single quadratic tet element (type 118)
        elem_ids = np.array([1], dtype=np.uint64)
        elem_types = np.array([118], dtype=np.uint8)
        elem_colors = np.array([1], dtype=np.uint16)
        connectivity = np.array([np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint64)], dtype=object)
        
        elements = element_array(
            id=elem_ids,
            type=elem_types,
            color=elem_colors,
            connectivity=connectivity
        )
        
        return nodes, elements

    def create_wedge_mesh(self):
        """Create a linear wedge mesh."""
        # 6 nodes for wedge
        node_ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)
        coordinates = np.array([
            [0.0, 0.0, 0.0],  # 1
            [1.0, 0.0, 0.0],  # 2
            [0.5, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [0.5, 1.0, 1.0],  # 6
        ], dtype=np.float64)
        
        nodes = node_array(id=node_ids, coordinate=coordinates)
        
        # Single wedge element (type 112)
        elem_ids = np.array([1], dtype=np.uint64)
        elem_types = np.array([112], dtype=np.uint8)
        elem_colors = np.array([1], dtype=np.uint16)
        connectivity = np.array([np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)], dtype=object)
        
        elements = element_array(
            id=elem_ids,
            type=elem_types,
            color=elem_colors,
            connectivity=connectivity
        )
        
        return nodes, elements

    def create_shell_tri_mesh(self):
        """Create a thin shell triangle mesh."""
        # 3 nodes for triangle
        node_ids = np.array([1, 2, 3], dtype=np.uint64)
        coordinates = np.array([
            [0.0, 0.0, 0.0],  # 1
            [1.0, 0.0, 0.0],  # 2
            [0.5, 1.0, 0.0],  # 3
        ], dtype=np.float64)
        
        nodes = node_array(id=node_ids, coordinate=coordinates)
        
        # Single thin shell triangle (type 91)
        elem_ids = np.array([1], dtype=np.uint64)
        elem_types = np.array([91], dtype=np.uint8)
        elem_colors = np.array([1], dtype=np.uint16)
        connectivity = np.array([np.array([1, 2, 3], dtype=np.uint64)], dtype=object)
        
        elements = element_array(
            id=elem_ids,
            type=elem_types,
            color=elem_colors,
            connectivity=connectivity
        )
        
        return nodes, elements

    def create_shell_quad_mesh(self):
        """Create a plate quadrilateral mesh."""
        # 4 nodes for quad
        node_ids = np.array([1, 2, 3, 4], dtype=np.uint64)
        coordinates = np.array([
            [0.0, 0.0, 0.0],  # 1
            [1.0, 0.0, 0.0],  # 2
            [1.0, 1.0, 0.0],  # 3
            [0.0, 1.0, 0.0],  # 4
        ], dtype=np.float64)
        
        nodes = node_array(id=node_ids, coordinate=coordinates)
        
        # Single plate quad element (type 64)
        elem_ids = np.array([1], dtype=np.uint64)
        elem_types = np.array([64], dtype=np.uint8)
        elem_colors = np.array([1], dtype=np.uint16)
        connectivity = np.array([np.array([1, 2, 3, 4], dtype=np.uint64)], dtype=object)
        
        elements = element_array(
            id=elem_ids,
            type=elem_types,
            color=elem_colors,
            connectivity=connectivity
        )
        
        return nodes, elements

    def create_quadratic_shell_mesh(self):
        """Create a quadratic shell triangle with midside nodes."""
        # 6 nodes: 3 corners + 3 midside
        node_ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)
        coordinates = np.array([
            [0.0, 0.0, 0.0],  # 1 - corner
            [1.0, 0.0, 0.0],  # 2 - corner
            [0.5, 1.0, 0.0],  # 3 - corner
            [0.5, 0.0, 0.0],  # 4 - midside 1-2
            [0.75, 0.5, 0.0], # 5 - midside 2-3
            [0.25, 0.5, 0.0], # 6 - midside 3-1
        ], dtype=np.float64)
        
        nodes = node_array(id=node_ids, coordinate=coordinates)
        
        # Single parabolic thin shell triangle (type 92)
        elem_ids = np.array([1], dtype=np.uint64)
        elem_types = np.array([92], dtype=np.uint8)
        elem_colors = np.array([1], dtype=np.uint16)
        connectivity = np.array([np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)], dtype=object)
        
        elements = element_array(
            id=elem_ids,
            type=elem_types,
            color=elem_colors,
            connectivity=connectivity
        )
        
        return nodes, elements

    def create_thick_shell_mesh(self):
        """Create a thick shell wedge mesh."""
        # 6 nodes for thick shell wedge
        node_ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)
        coordinates = np.array([
            [0.0, 0.0, 0.0],  # 1
            [1.0, 0.0, 0.0],  # 2
            [0.5, 1.0, 0.0],  # 3
            [0.0, 0.0, 0.1],  # 4 (small thickness)
            [1.0, 0.0, 0.1],  # 5
            [0.5, 1.0, 0.1],  # 6
        ], dtype=np.float64)
        
        nodes = node_array(id=node_ids, coordinate=coordinates)
        
        # Single thick shell wedge (type 101)
        elem_ids = np.array([1], dtype=np.uint64)
        elem_types = np.array([101], dtype=np.uint8)
        elem_colors = np.array([1], dtype=np.uint16)
        connectivity = np.array([np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)], dtype=object)
        
        elements = element_array(
            id=elem_ids,
            type=elem_types,
            color=elem_colors,
            connectivity=connectivity
        )
        
        return nodes, elements

    def test_quadratic_tet_extraction(self):
        """Test extraction from quadratic tetrahedron (uses only corner nodes)."""
        nodes, elements = self.create_quadratic_tet_mesh()
        extractor = BoundaryNodeExtractor(nodes, elements)
        node_out = extractor.get_node_out()
        elem_out = extractor.get_elem_out()

        # Should use only corner nodes (1,2,3,4) for face extraction
        assert len(elem_out) == 4  # 4 triangular faces
        assert all(elem_out['type'] == 41)  # triangles
        
        # All faces should use only corner node IDs
        face_nodes = set()
        for conn in elem_out['connectivity']:
            face_nodes.update(conn)
        assert face_nodes.issubset({1, 2, 3, 4})  # only corner nodes

    def test_wedge_extraction(self):
        """Test extraction from linear wedge."""
        nodes, elements = self.create_wedge_mesh()
        extractor = BoundaryNodeExtractor(nodes, elements)
        node_out = extractor.get_node_out()
        elem_out = extractor.get_elem_out()

        assert len(node_out) == 6
        assert set(node_out['id']) == {1, 2, 3, 4, 5, 6}

        # Should have 5 faces: 2 triangles + 3 quads
        assert len(elem_out) == 5
        
        # Count triangular and quad faces
        tri_faces = sum(1 for conn in elem_out['connectivity'] if len(conn) == 3)
        quad_faces = sum(1 for conn in elem_out['connectivity'] if len(conn) == 4)
        assert tri_faces == 2
        assert quad_faces == 3

    def test_shell_triangle_extraction(self):
        """Test extraction from thin shell triangle (2D element)."""
        nodes, elements = self.create_shell_tri_mesh()
        extractor = BoundaryNodeExtractor(nodes, elements)
        node_out = extractor.get_node_out()
        elem_out = extractor.get_elem_out()

        # For 2D elements, should pass through as boundary
        assert len(node_out) == 3
        assert set(node_out['id']) == {1, 2, 3}
        
        assert len(elem_out) == 1
        assert elem_out['type'][0] == 41  # triangle
        assert set(elem_out['connectivity'][0]) == {1, 2, 3}

    def test_shell_quad_extraction(self):
        """Test extraction from plate quadrilateral (2D element)."""
        nodes, elements = self.create_shell_quad_mesh()
        extractor = BoundaryNodeExtractor(nodes, elements)
        node_out = extractor.get_node_out()
        elem_out = extractor.get_elem_out()

        # For 2D elements, should pass through as boundary
        assert len(node_out) == 4
        assert set(node_out['id']) == {1, 2, 3, 4}
        
        assert len(elem_out) == 1
        assert elem_out['type'][0] == 44  # quad
        assert set(elem_out['connectivity'][0]) == {1, 2, 3, 4}

    def test_quadratic_shell_extraction(self):
        """Test extraction from quadratic shell (uses only corner nodes)."""
        nodes, elements = self.create_quadratic_shell_mesh()
        extractor = BoundaryNodeExtractor(nodes, elements)
        node_out = extractor.get_node_out()
        elem_out = extractor.get_elem_out()

        # Should use only corner nodes for 2D element
        assert len(elem_out) == 1
        assert elem_out['type'][0] == 41  # triangle
        assert set(elem_out['connectivity'][0]) == {1, 2, 3}  # only corner nodes
        
        # Node output should contain corner nodes
        assert {1, 2, 3}.issubset(set(node_out['id']))

    def test_thick_shell_extraction(self):
        """Test extraction from thick shell wedge (3D element)."""
        nodes, elements = self.create_thick_shell_mesh()
        extractor = BoundaryNodeExtractor(nodes, elements)
        node_out = extractor.get_node_out()
        elem_out = extractor.get_elem_out()

        assert len(node_out) == 6
        assert set(node_out['id']) == {1, 2, 3, 4, 5, 6}

        # Should have 5 faces like regular wedge
        assert len(elem_out) == 5
        
        # Count triangular and quad faces
        tri_faces = sum(1 for conn in elem_out['connectivity'] if len(conn) == 3)
        quad_faces = sum(1 for conn in elem_out['connectivity'] if len(conn) == 4)
        assert tri_faces == 2
        assert quad_faces == 3

    def test_mixed_element_types(self):
        """Test extraction from mixed element types."""
        # Create a mesh with both solid and shell elements
        nodes = node_array(
            id=np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64),
            coordinate=np.array([
                [0.0, 0.0, 0.0],  # 1
                [1.0, 0.0, 0.0],  # 2
                [0.0, 1.0, 0.0],  # 3
                [0.0, 0.0, 1.0],  # 4
                [2.0, 0.0, 0.0],  # 5
                [2.0, 1.0, 0.0],  # 6
            ], dtype=np.float64)
        )
        
        # Mix of tet (solid) and shell triangle
        elem_ids = np.array([1, 2], dtype=np.uint64)
        elem_types = np.array([111, 91], dtype=np.uint8)  # solid tet + thin shell tri
        elem_colors = np.array([1, 2], dtype=np.uint16)
        connectivity = np.array([
            np.array([1, 2, 3, 4], dtype=np.uint64),  # tet
            np.array([2, 5, 6], dtype=np.uint64),     # shell tri
        ], dtype=object)
        
        elements = element_array(
            id=elem_ids,
            type=elem_types,
            color=elem_colors,
            connectivity=connectivity
        )
        
        extractor = BoundaryNodeExtractor(nodes, elements)
        node_out = extractor.get_node_out()
        elem_out = extractor.get_elem_out()
        
        # Should extract faces from tet + pass through shell
        assert len(elem_out) == 5  # 4 tet faces + 1 shell
        
        # All output should be triangles since tet faces are triangles and shell is triangle
        assert all(elem_out['type'] == 41)

    def test_various_element_types_individually(self):
        """Test various specific element types that should be supported."""
        test_cases = [
            # (element_type, min_nodes, expected_behavior)
            (118, 10, "quadratic_tet"),   # Quadratic tet
            (113, 15, "quadratic_wedge"), # Quadratic wedge  
            (116, 20, "quadratic_hex"),   # Quadratic hex
            (62, 6, "shell_tri"),         # Parabolic plate tri
            (65, 8, "shell_quad"),        # Parabolic plate quad
            (102, 15, "thick_shell"),     # Parabolic thick shell wedge
        ]
        
        for elem_type, min_nodes, behavior in test_cases:
            # Create minimal nodes
            node_ids = np.arange(1, min_nodes + 1, dtype=np.uint64)
            coordinates = np.random.rand(min_nodes, 3).astype(np.float64)
            nodes = node_array(id=node_ids, coordinate=coordinates)
            
            # Create single element
            elem_ids = np.array([1], dtype=np.uint64)
            elem_types = np.array([elem_type], dtype=np.uint8)
            elem_colors = np.array([1], dtype=np.uint16)
            connectivity = np.array([node_ids], dtype=object)
            
            elements = element_array(
                id=elem_ids,
                type=elem_types,
                color=elem_colors,
                connectivity=connectivity
            )
            
            # Should not raise error
            extractor = BoundaryNodeExtractor(nodes, elements)
            node_out = extractor.get_node_out()
            elem_out = extractor.get_elem_out()
            
            # Basic validity checks
            assert len(node_out) > 0
            assert len(elem_out) > 0
            assert all(t in [41, 44] for t in elem_out['type'])  # only tri/quad output


