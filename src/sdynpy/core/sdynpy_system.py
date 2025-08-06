# -*- coding: utf-8 -*-
"""
Defines a system of matrices representing a structure.

The System consists of mass, stiffness, and (if necessary) damping matrices.
The System also contains a CoordinateArray defining the degrees of freedom of
the System, as well as a transformation that takes the System from its internal
state degrees of freedom to physical degrees of freedom.

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
from .sdynpy_coordinate import CoordinateArray, from_nodelist, outer_product, coordinate_array
from ..fem.sdynpy_beam import beamkm, rect_beam_props
from ..fem.sdynpy_exodus import Exodus, ExodusInMemory, reduce_exodus_to_surfaces
from ..signal_processing import frf as spfrf
from ..signal_processing import generator
from scipy.linalg import eigh, block_diag, null_space, eig
from scipy.signal import lsim, StateSpace, resample, butter, filtfilt
import copy
import netCDF4 as nc4
import matplotlib.pyplot as plt
import warnings
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import ArpackNoConvergence


class System:
    """Matrix Equations representing a Structural Dynamics System"""

    def __init__(self, coordinate: CoordinateArray, mass, stiffness, damping=None,
                 transformation=None):
        """
        Create a system representation including mass, stiffness, damping, and
        transformation matrices.

        Parameters
        ----------
        coordinate : CoordinateArray
            Physical degrees of freedom in the system.
        mass : np.ndarray
            2D array consisting of the mass matrix of the system
        stiffness : np.ndarray
            2D array consisting of the stiffness matrix of the system
        damping : np.ndarray, optional
            2D array consisting of the damping matrix of the system.  If not
            specified, the damping will be zero.
        transformation : np.ndarray, optional
            A transformation between internal "state" degrees of freedom and
            the physical degrees of freedom defined in `coordinate`. The
            default transformation is the identity matrix.

        Raises
        ------
        ValueError
            If inputs are improperly sized

        Returns
        -------
        None.

        """
        mass = np.atleast_2d(np.array(mass))
        stiffness = np.atleast_2d(np.array(stiffness))
        if not mass.shape == stiffness.shape:
            raise ValueError('Mass and Stiffness matrices must be the same shape')
        if not mass.ndim == 2 or (mass.shape[0] != mass.shape[1]):
            raise ValueError('Mass should be a 2D, square array')
        if not stiffness.ndim == 2 or (stiffness.shape[0] != stiffness.shape[1]):
            raise ValueError('Stiffness should be a 2D, square array')
        if damping is None:
            damping = np.zeros(stiffness.shape)
        else:
            damping = np.atleast_2d(np.array(damping))
        if not damping.shape == stiffness.shape:
            raise ValueError('Damping and Stiffness matrices must be the same shape')
        if not damping.ndim == 2 or (damping.shape[0] != damping.shape[1]):
            raise ValueError('Damping should be a 2D, square array')
        if transformation is None:
            transformation = np.eye(mass.shape[0])
        else:
            transformation = np.atleast_2d(np.array(transformation))
        if not transformation.ndim == 2:
            raise ValueError('transformation must be 2D')
        if not transformation.shape[-1] == mass.shape[0]:
            raise ValueError(
                'transformation must have number of columns equal to the number of rows in the mass matrix')
        coordinate = np.atleast_1d(coordinate)
        if not isinstance(coordinate, CoordinateArray):
            raise ValueError('coordinate must be a CoordinateArray object')
        if not coordinate.ndim == 1 or coordinate.shape[0] != transformation.shape[0]:
            raise ValueError(
                'coordinate must be 1D and have the same size as transformation.shape[0] or mass.shape[0] if no transformation is specified')
        # Check symmetry
        if not np.allclose(mass, mass.T):
            raise ValueError('mass matrix must be symmetric')
        if not np.allclose(stiffness, stiffness.T, atol=1e-6*stiffness.max()):
            raise ValueError('stiffness matrix must be symmetric')
        if not np.allclose(damping, damping.T):
            raise ValueError('damping matrix must be symmetric')

        self._coordinate = coordinate
        self._mass = mass
        self._stiffness = stiffness
        self._damping = damping
        self._transformation = transformation

    def __repr__(self):
        return 'System with {:} DoFs ({:} internal DoFs)'.format(self.ndof_transformed, self.ndof)

    def spy(self, subplots_kwargs={'figsize': (10, 3)}, spy_kwargs={}):
        """
        Plot the structure of the system's matrices

        Parameters
        ----------
        subplots_kwargs : dict, optional
            Default arguments passed to `matplotlib.pyplot`'s `subplots` function.
            The default is {'figsize':(10,3)}.
        spy_kwargs : dict, optional
            Default arguments passed to `matplotlib.pyplot`'s 'spy' function.
            The default is {}.

        Returns
        -------
        ax : Axes
            Axes on which the subplots are defined.

        """
        fig, ax = plt.subplots(1, 3, squeeze=True, **subplots_kwargs)
        ax[0].spy(abs(self.transformation), **spy_kwargs)
        ax[1].spy(abs(self.mass) + abs(self.stiffness) + abs(self.damping), **spy_kwargs)
        ax[2].spy(abs(self.transformation.T), **spy_kwargs)
        ax[0].set_title('Output Transformation')
        ax[1].set_title('Internal State Matrices')
        ax[2].set_title('Input Transformation')
        fig.tight_layout()
        # Trying to figure out how to scale the subfigures identically...
        # plt.pause(0.01)
        # # Now adjust the sizes of the plots
        # sizes = []
        # for a in ax:
        #     bbox = a.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #     width, height = bbox.width, bbox.height
        #     sizes.append((width, height))
        # scales = [sizes[0][0],sizes[1][1],sizes[2][1]]
        # plt.pause(0.01)
        # scales = [scale/min(scales) for scale in scales]
        # for a,scale in zip(ax,scales):
        #     position = a.get_position()
        #     center_x = (position.xmax + position.xmin)/2
        #     center_y = (position.ymax + position.ymin)/2
        #     new_left = center_x - (center_x - position.xmin)/2
        #     new_bottom = center_y - (center_y - position.ymin)/2
        #     a.set_position([new_left,new_bottom,position.width/scale,position.height/scale])
        return ax

    @property
    def transformation(self):
        """Get or set the transformation matrix"""
        return self._transformation

    @transformation.setter
    def transformation(self, value):
        """Set the transformation matrix"""
        if not value.ndim == 2:
            raise ValueError('transformation must be 2D')
        if not value.shape[-1] == self.mass.shape[0]:
            raise ValueError(
                'transformation must have number of columns equal to the number of rows in the mass matrix')
        self._transformation = value

    @property
    def coordinate(self):
        """Get or set the degrees of freedom in the system"""
        return self._coordinate

    @coordinate.setter
    def coordinate(self, value):
        """Set the degrees of freedom in the system"""
        if not isinstance(value, CoordinateArray):
            raise ValueError('coordinate must be a CoordinateArray object')
        if not value.ndim == 1 or value.shape[0] != self.mass.shape[0]:
            raise ValueError('coordinate must be 1D and have the same size as mass.shape[0]')
        self._coordinate = value

    @property
    def mass(self):
        """Get or set the mass matrix of the system"""
        return self._mass

    @mass.setter
    def mass(self, value):
        """Set the mass matrix of the system"""
        if not value.shape == self.stiffness.shape:
            raise ValueError('Mass and Stiffness matrices must be the same shape')
        if not value.ndim == 2 or (value.shape[0] != value.shape[1]):
            raise ValueError('Mass should be a 2D, square array')
        if not np.allclose(value, value.T):
            raise ValueError('mass matrix must be symmetric')
        self._mass = value

    @property
    def stiffness(self):
        """Get or set the stffness matrix of the system"""
        return self._stiffness

    @stiffness.setter
    def stiffness(self, value):
        """Set the stiffness matrix of the system"""
        if not value.shape == self.mass.shape:
            raise ValueError('Mass and Stiffness matrices must be the same shape')
        if not value.ndim == 2 or (value.shape[0] != value.shape[1]):
            raise ValueError('Stiffness should be a 2D, square array')
        if not np.allclose(value, value.T):
            raise ValueError('stiffness matrix must be symmetric')
        self._stiffness = value

    @property
    def damping(self):
        """Get or set the damping matrix of the system"""
        return self._damping

    @damping.setter
    def damping(self, value):
        """Set the damping matrix of the system"""
        if not value.shape == self.stiffness.shape:
            raise ValueError('damping and Stiffness matrices must be the same shape')
        if not value.ndim == 2 or (value.shape[0] != value.shape[1]):
            raise ValueError('damping should be a 2D, square array')
        if not np.allclose(value, value.T):
            raise ValueError('damping matrix must be symmetric')
        self._damping = value

    M = mass
    K = stiffness
    C = damping

    @property
    def ndof(self):
        """Get the number of internal degrees of freedom of the system"""
        return self.mass.shape[0]

    @property
    def ndof_transformed(self):
        """Get the number of physical degrees of freedom of the system"""
        return self.transformation.shape[0]

    def to_state_space(self, output_displacement=True, output_velocity=True, output_acceleration=True, output_force=True,
                       response_coordinates=None, input_coordinates=None):
        """
        Compute the state space representation of the system

        Parameters
        ----------
        output_displacement : bool, optional
            Provide rows in the output and feedforward matrices corresponding to
            displacement outputs.  The default is True.
        output_velocity : bool, optional
            Provide rows in the output and feedforward matrices corresponding to
            velocity outputs.  The default is True.
        output_acceleration : bool, optional
            Provide rows in the output and feedforward matrices corresponding to
            acceleration outputs.  The default is True.
        output_force : bool, optional
            Provide rows in the output and feedforward matrices corresponding to
            force outputs.  The default is True.

        Returns
        -------
        A_state : np.ndarray
            The state matrix
        B_state : np.ndarray
            The input matrix
        C_state : np.ndarray
            The output matrix.
        D_state : np.ndarray
            The feedforward matrix.

        """
        ndofs = self.ndof
        M = self.M
        K = self.K
        C = self.C
        if response_coordinates is None:
            phi_response = self.transformation
        elif isinstance(response_coordinates, str) and response_coordinates == 'state':
            # If you want the state variables back, the transformation is the
            # identity matrix.
            phi_response = np.eye(self.transformation.shape[-1])
        else:
            phi_response = self.transformation_matrix_at_coordinates(response_coordinates)
        if input_coordinates is None:
            phi_input = self.transformation
        else:
            phi_input = self.transformation_matrix_at_coordinates(input_coordinates)
        tdofs_response = phi_response.shape[0]
        tdofs_input = phi_input.shape[0]
        # A = [[     0,     I],
        #      [M^-1*K,M^-1*C]]

        A_state = np.block([[np.zeros((ndofs, ndofs)), np.eye(ndofs)],
                            [-np.linalg.solve(M, K), -np.linalg.solve(M, C)]])

        # B = [[     0],
        #      [  M^-1]]

        B_state = np.block([[np.zeros((ndofs, tdofs_input))],
                            [np.linalg.solve(M, phi_input.T)]])

        # C = [[     I,     0],   # Displacements
        #      [     0,     I],   # Velocities
        #      [M^-1*K,M^-1*C],   # Accelerations
        #      [     0,     0]]   # Forces

        C_state = np.block([[phi_response, np.zeros((tdofs_response, ndofs))],
                            [np.zeros((tdofs_response, ndofs)), phi_response],
                            [-phi_response @ np.linalg.solve(M, K), -phi_response @ np.linalg.solve(M, C)],
                            [np.zeros((tdofs_input, ndofs)), np.zeros((tdofs_input, ndofs))]])

        # D = [[     0],   # Displacements
        #      [     0],   # Velocities
        #      [  M^-1],   # Accelerations
        #      [     I]]   # Forces

        D_state = np.block([[np.zeros((tdofs_response, tdofs_input))],
                            [np.zeros((tdofs_response, tdofs_input))],
                            [phi_response @ np.linalg.solve(M, phi_input.T)],
                            [np.eye(tdofs_input)]])
        displacement_indices = np.arange(tdofs_response)
        velocity_indices = np.arange(tdofs_response) + tdofs_response
        acceleration_indices = np.arange(tdofs_response) + 2 * tdofs_response
        force_indices = np.arange(tdofs_input) + 3 * tdofs_response
        output_indices = np.zeros(3 * tdofs_response + tdofs_input, dtype=bool)
        if output_displacement:
            output_indices[displacement_indices] = True
        if output_velocity:
            output_indices[velocity_indices] = True
        if output_acceleration:
            output_indices[acceleration_indices] = True
        if output_force:
            output_indices[force_indices] = True
        C_state = C_state[output_indices]
        D_state = D_state[output_indices]
        return A_state, B_state, C_state, D_state

    def time_integrate(self, forces, dt=None, responses=None, references=None,
                       displacement_derivative=2, initial_state=None,
                       integration_oversample=1):
        """
        Integrate a system to produce responses to an excitation

        Parameters
        ----------
        forces : np.ndarray or TimeHistoryArray
            The forces applied to the system, which should be a signal with
            time-step `dt`.  If a `TimeHistoryArray` is passed, then `dt` and
            `references` will be taken from the `TimeHistoryArray`, and the
            arguments will be ignored.
        dt : float, optional
            The timestep used for the integration.  Must be specified if `forces`
            is a ndarray.  If `forces` is a TimeHistoryArray, then this argument
            is ignored.
        responses : CoordinateArray, optional
            Coordinates at which responses are desired. The default is all
            responses.
        references : CoordinateArray, optional
            Coordinates at which responses are input. The default is all
            references.  Must be the same size as the number of rows in the
            `forces` array.  If `forces` is a `TimeHistoryArray`, then this
            argument is ignored.
        displacement_derivative : int, optional
            The derivative of the displacement that the output will be
            represented as.  A derivative of 0 means displacements will be
            returned, a derivative of 1 will mean velocities will be returned,
            and a derivative of 2 will mean accelerations will be returned.
            The default is 2, which returns accelerations.
        initial_state : np.ndarray, optional
            The initial conditions of the integration. The default is zero
            displacement and zero velocity.
        integration_oversample : int
            The amount of oversampling that will be applied to the force by
            zero-padding the fft

        Returns
        -------
        response_array : TimeHistoryArray
            The responses of the system to the forces applied
        reference_array : TimeHistoryArray
            The forces applied to the system as a TimeHistoryArray

        """
        from .sdynpy_data import data_array, FunctionTypes, TimeHistoryArray
        if isinstance(forces, TimeHistoryArray):
            dt = forces.abscissa_spacing
            references = forces.coordinate[..., 0]
            forces = forces.ordinate
        else:
            if dt is None:
                raise ValueError('`dt` must be specified if `forces` is not a `TimeHistoryArray`')
        if responses is not None and not isinstance(responses, str):
            responses = np.atleast_1d(responses)
        if references is not None:
            references = np.atleast_1d(references)
        A, B, C, D = self.to_state_space(displacement_derivative == 0,
                                         displacement_derivative == 1,
                                         displacement_derivative == 2,
                                         False,
                                         responses, references
                                         )
        forces = np.atleast_2d(forces)
        times = np.arange(forces.shape[-1]) * dt
        linear_system = StateSpace(A, B, C, D)
        if integration_oversample != 1:
            forces, times = resample(forces, len(times) * integration_oversample, times, axis=-1)
        times_out, time_response, x_out = lsim(linear_system, forces.T, times, initial_state)
        if time_response.ndim == 1:
            time_response = time_response[:, np.newaxis]
        if responses is None:
            response_coordinate = self.coordinate.copy()
        elif isinstance(responses, str) and responses == 'state':
            response_coordinate = coordinate_array(np.arange(self.ndof), 0)
        else:
            response_coordinate = responses
        response_array = data_array(FunctionTypes.TIME_RESPONSE,
                                    times, time_response.T,
                                    response_coordinate[:, np.newaxis])
        reference_coordinate = self.coordinate.copy() if references is None else references
        reference_array = data_array(FunctionTypes.TIME_RESPONSE,
                                     times, np.atleast_2d(forces),
                                     reference_coordinate[:, np.newaxis])
        if integration_oversample != 1:
            response_array = response_array.extract_elements(
                slice(None, None, integration_oversample))
            reference_array = reference_array.extract_elements(
                slice(None, None, integration_oversample))
        return response_array, reference_array

    def eigensolution(self, num_modes=None, maximum_frequency=None, complex_modes=False, return_shape=True):
        """
        Computes the eigensolution of the system

        Parameters
        ----------
        num_modes : int, optional
            The number of modes of the system to compute. The default is to
            compute all the modes.
        maximum_frequency : float, optional
            The maximum frequency to which modes will be computed.
            The default is to compute all the modes.
        complex_modes : bool, optional
            Whether or not complex modes are computed. The default is False.
        return_shape : bool, optional
            Specifies whether or not to return a `ShapeArray` (True) or a reduced
            `System` (False). The default is True.

        Raises
        ------
        NotImplementedError
            Raised if complex modes are specified.

        Returns
        -------
        System or ShapeArray
            If `return_shape` is True, the a ShapeArray will be returned.  If
            `return_shape` is False, a reduced system will be returned.

        """
        if complex_modes is False:
            if num_modes is not None:
                num_modes = [0, int(num_modes) - 1]
            if maximum_frequency is not None:
                maximum_frequency = (2 * np.pi * maximum_frequency)**2
                maximum_frequency = [-maximum_frequency, maximum_frequency]  # Convert to eigenvalue
            lam, phi = eigh(self.K, self.M, subset_by_index=num_modes,
                            subset_by_value=maximum_frequency)
            # Mass normalize the mode shapes
            lam[lam < 0] = 0
            freq = np.sqrt(lam) / (2 * np.pi)
            normalized_mass = np.diag(phi.T @ self.M @ phi)
            phi /= np.sqrt(normalized_mass)
            # Ignore divide by zero if the frequency is zero
            with np.errstate(divide='ignore', invalid='ignore'):
                damping = np.diag(phi.T @ self.C @ phi) / (2 * (2 * np.pi * freq))
            damping[np.isnan(damping)] = 0.0
            damping[np.isinf(damping)] = 0.0
            # Add in the transformation to get back to physical dofs
            phi = self.transformation @ phi
            if return_shape:
                from .sdynpy_shape import shape_array
                return shape_array(self.coordinate, phi.T, freq, damping)
            else:
                return System(self.coordinate, np.eye(freq.size), np.diag((2 * np.pi * freq)**2), np.diag(2 * (2 * np.pi * freq) * damping), phi)
        else:
            if self.ndof > 1000:
                warnings.warn('The complex mode implementation currently computes all eigenvalues and eigenvectors, which may take a long time for large systems.')
            # For convenience, assign a zeros matrix
            Z = np.zeros(self.M.shape)
            A = np.block([[     Z, self.M],
                          [self.M, self.C]])
            B = np.block([[-self.M,      Z],
                          [      Z, self.K]])
            lam, E = eig(-B,A)
            # Sort the eigenvalues such that they are increasing in frequency
            isort = np.argsort(np.abs(lam))
            lam = lam[isort]
            E = E[:,isort]
            # Eigenvalues will be in complex conjugate pairs.  Let's only keep
            # the ones that are greater than zero.
            keep = lam.imag > 0
            lam = lam[keep]
            E = E[:,keep]
            if A.shape[0]//2 != lam.size:
                warnings.warn('The complex mode implementation currently does not do well with rigid body modes (0 Hz frequencies).  Compute them from a real-modes solution then transform to complex.')
            # Cull values we don't want
            if num_modes is not None:
                lam = lam[:num_modes]
                E = E[:,:num_modes]
            if maximum_frequency is not None:
                keep = np.abs(lam) <= maximum_frequency*2*np.pi
                lam = lam[keep]
                E = E[:,keep]
            # Mass normalize the mode shapes
            E = E/np.sqrt(np.einsum('ji,jk,ki->i',E,A,E))
            # Find repeated eigenvalues where the eigenvectors are not orthogonal
            # TODO: Might have to do some orthogonalization for repeated
            # eigenvalues
            # A_modal = np.einsum('ji,jk,kl->il',E,A,E)
            # Extract just the displacement partition
            psi = E[E.shape[0]//2:,:]
            # Add in a transformation to get to physical dofs
            psi_t = self.transformation @ psi
            frequency = np.abs(lam)/(2*np.pi)
            damping = -np.real(lam)/np.abs(lam)
            if return_shape:
                from .sdynpy_shape import shape_array
                return shape_array(self.coordinate, psi.T, frequency, damping)
            else:
                warnings.warn('Complex Modes will in general not diagonalize the system M, C, and K, matrices.')
                return System(self.coordinate,
                              psi.T@self.M@psi,
                              psi.T@self.K@psi.T,
                              psi.T@self.C@psi.T,
                              psi_t)
            

    def transformation_shapes(self, shape_indices=None):
        from .sdynpy_shape import shape_array
        if shape_indices is None:
            shape_indices = slice(None)
        shape_matrix = self.transformation[:, shape_indices]
        return shape_array(self.coordinate, shape_matrix.T,
                           frequency=0, damping=0)

    def remove_transformation(self):
        return System(coordinate_array(np.arange(self.ndof)+1,0),
                      self.mass.copy(),self.stiffness.copy(),self.damping.copy())

    def frequency_response(self, frequencies, responses=None, references=None,
                           displacement_derivative=0):
        """
        Computes frequency response functions at the specified frequency lines.

        Parameters
        ----------
        frequencies : ndarray
            A 1D array of frequencies.
        responses : CoordinateArray, optional
            A set of coordinates to compute responses. The default is to
            create responses at all coordinates.
        references : CoordinateArray, optional
            A set of coordinates to use as inputs. The default is to use all
            coordinates as inputs.
        displacement_derivative : int, optional
            The number of derivatives to apply to the response. The default is
            0, which corresponds to displacement.  1 would be the first
            derivative, velocity, and 2 would be the second derivative,
            acceleration.

        Returns
        -------
        frf : TransferFunctionArray
            A TransferFunctionArray containing the frequency response function
            for the system at the specified input and output degrees of freedom.

        """
        H = spfrf.sysmat2frf(frequencies, self.M, self.C, self.K)
        H = (1j * (2 * np.pi * frequencies[:, np.newaxis, np.newaxis]))**displacement_derivative * H
        # Apply transformations
        if responses is None:
            output_transform = self.transformation
            output_coordinates = self.coordinate
        else:
            responses = np.atleast_1d(responses)
            output_transform = self.transformation_matrix_at_coordinates(responses)
            output_coordinates = responses
        if references is None:
            input_transform = self.transformation
            input_coordinates = self.coordinate
        else:
            references = np.atleast_1d(references)
            input_transform = self.transformation_matrix_at_coordinates(references)
            input_coordinates = references
        H = output_transform @ H @ input_transform.T
        # Put it into a transfer function array
        from .sdynpy_data import data_array, FunctionTypes
        frf = data_array(FunctionTypes.FREQUENCY_RESPONSE_FUNCTION, frequencies,
                         np.moveaxis(H, 0, -1),
                         outer_product(output_coordinates, input_coordinates))
        return frf

    def assign_modal_damping(self, damping_ratios):
        """
        Assigns a damping matrix to the system that results in equivalent
        modal damping

        Parameters
        ----------
        damping_ratios : ndarray
            An array of damping values to assign to the system

        Returns
        -------
        None.

        """
        damping_ratios = np.array(damping_ratios)
        if damping_ratios.ndim == 1:
            shapes = self.eigensolution(num_modes=damping_ratios.size)
        else:
            shapes = self.eigensolution()
        shapes.damping = damping_ratios
        # Compute the damping matrix
        modal_system = shapes.system()
        shape_pinv = np.linalg.pinv(modal_system.transformation.T)
        full_damping_matrix = shape_pinv@modal_system.damping@shape_pinv.T
        self.damping[:] = full_damping_matrix

    def save(self, filename):
        """
        Saves the system to a file

        Parameters
        ----------
        filename : str
            Name of the file in which the system will be saved.

        Returns
        -------
        None.

        """
        np.savez(filename, mass=self.mass, stiffness=self.stiffness, damping=self.damping,
                 transformation=self.transformation, coordinate=self.coordinate.view(np.ndarray))

    @classmethod
    def load(cls, filename):
        """
        Load a system from a file

        Parameters
        ----------
        filename : str
            Name of the file from which the system will be loaded.

        Returns
        -------
        System
            A system consisting of the mass, stiffness, damping, and transformation
            in the file

        """
        data = np.load(filename)
        return cls(data['coordinate'].view(CoordinateArray), data['mass'],
                   data['stiffness'], data['damping'], data['transformation'])

    def __neg__(self):
        new_system = copy.deepcopy(self)
        new_system.mass *= -1
        new_system.stiffness *= -1
        new_system.damping *= -1
        return new_system

    @classmethod
    def concatenate(cls, systems, coordinate_node_offset=0):
        """
        Combine multiple systems together

        Parameters
        ----------
        systems : iterable of System objects
            Iterable of Systems that will be concatenated.  Matrices will be
            assembled in block diagonal format
        coordinate_node_offset : int, optional
            Offset applied to the coordinates so the nodes do not overlap.
            The default is 0.

        Returns
        -------
        System
            A system consisting of the combintation of the provided systems.

        """
        coordinates = [system.coordinate.copy() for system in systems]
        if coordinate_node_offset != 0:
            for i in range(len(coordinates)):
                coordinates[i].node += coordinate_node_offset * (i + 1)
        all_coordinates = np.concatenate(coordinates)
        return cls(all_coordinates,
                   block_diag(*[system.mass for system in systems]),
                   block_diag(*[system.stiffness for system in systems]),
                   block_diag(*[system.damping for system in systems]),
                   block_diag(*[system.transformation for system in systems]))

    @classmethod
    def substructure_by_position(cls, systems, geometries, distance_threshold=1e-8, rcond=None):
        """
        Applies constraints to systems by constraining colocated nodes together

        Parameters
        ----------
        systems : iterable of System objects
            A set of systems that will be combined and constrained together
        geometries : iterable of Geometry objects
            A set of geometries that will be combined together
        distance_threshold : float, optional
            The distance between nodes that are considered colocated.
            The default is 1e-8.
        rcond : float, optional
            Condition number to use in the nullspace calculation on
            the constraint matrix. The default is None.

        Returns
        -------
        combined_system : System
            System consisting of constraining the input systems together.
        combined_geometry : Geometry
            Combined geometry of the new system

        """
        from .sdynpy_geometry import Geometry
        combined_geometry, node_offset = Geometry.overlay_geometries(
            geometries, return_node_id_offset=True)
        combined_system = cls.concatenate(systems, node_offset)
        global_coords = combined_geometry.global_node_coordinate()
        node_distances = np.linalg.norm(
            global_coords[:, np.newaxis, :] - global_coords[np.newaxis, :, :], axis=-1)
        # Find locations where the value is less than the tolerances, except for on the centerline
        node_pairs = [[combined_geometry.node.id[index] for index in pair]
                      for pair in zip(*np.where(node_distances < distance_threshold)) if pair[0] < pair[1]]
        # Find matching DoF pairs
        constraint_matrix = []
        for node1, node2 in node_pairs:
            # Find the dofs associated with each node
            system_1_dof_indices = np.where(combined_system.coordinate.node == node1)[0]
            system_1_dofs = combined_system.coordinate[system_1_dof_indices]
            system_1_transformation = combined_system.transformation[system_1_dof_indices]
            global_deflections_1 = combined_geometry.global_deflection(system_1_dofs)
            system_2_dof_indices = np.where(combined_system.coordinate.node == node2)[0]
            system_2_dofs = combined_system.coordinate[system_2_dof_indices]
            system_2_transformation = combined_system.transformation[system_2_dof_indices]
            global_deflections_2 = combined_geometry.global_deflection(system_2_dofs)
            # Split between translations and rotations
            translation_map_1 = np.where((abs(system_1_dofs.direction) <= 3)
                                         & (abs(system_1_dofs.direction) > 0))
            rotation_map_1 = np.where(abs(system_1_dofs.direction) > 3)
            translation_map_2 = np.where((abs(system_2_dofs.direction) <= 3)
                                         & (abs(system_2_dofs.direction) > 0))
            rotation_map_2 = np.where(abs(system_2_dofs.direction) > 3)
            neutral_map_1 = np.where(abs(system_1_dofs.direction) == 0)
            neutral_map_2 = np.where(abs(system_2_dofs.direction) == 0)
            # Do translations and rotations separately
            for map_1, map_2 in [[translation_map_1, translation_map_2],
                                 [rotation_map_1, rotation_map_2],
                                 [neutral_map_1, neutral_map_2]]:
                deflections_1 = global_deflections_1[map_1].T
                deflections_2 = global_deflections_2[map_2].T
                transform_1 = system_1_transformation[map_1]
                transform_2 = system_2_transformation[map_2]
                full_constraint = deflections_1 @ transform_1 - deflections_2 @ transform_2
                constraint_matrix.append(full_constraint)
        constraint_matrix = np.concatenate(constraint_matrix, axis=0)
        return combined_system.constrain(constraint_matrix, rcond), combined_geometry

    def constrain(self, constraint_matrix, rcond=None):
        """
        Apply a constraint matrix to the system

        Parameters
        ----------
        constraint_matrix : np.ndarray
            A matrix of constraints to apply to the structure (B matrix in
            substructuring literature)
        rcond : float, optional
            Condition tolerance for computing the nullspace. The default is None.

        Returns
        -------
        System
            Constrained system.

        """
        substructuring_transform_matrix = null_space(constraint_matrix, rcond)
        new_mass = substructuring_transform_matrix.T @ self.mass @ substructuring_transform_matrix
        new_stiffness = substructuring_transform_matrix.T @ self.stiffness @ substructuring_transform_matrix
        new_damping = substructuring_transform_matrix.T @ self.damping @ substructuring_transform_matrix
        new_transform = self.transformation @ substructuring_transform_matrix
        return System(self.coordinate, new_mass, new_stiffness, new_damping, new_transform)

    def transformation_matrix_at_coordinates(self, coordinates):
        """
        Return the transformation matrix at the specified coordinates

        Parameters
        ----------
        coordinates : CoordinateArray
            coordinates at which the transformation matrix will be computed.

        Raises
        ------
        ValueError
            Raised if duplicate coordinates are requested, or if coordinates that
            do not exist in the system are requested.

        Returns
        -------
        return_value : np.ndarray
            Portion of the transformation matrix corresponding to the
            coordinates input to the function.

        """
        consistent_arrays, shape_indices, request_indices = np.intersect1d(
            abs(self.coordinate), abs(coordinates), assume_unique=False, return_indices=True)
        # Make sure that all of the keys are actually in the consistent array matrix
        if consistent_arrays.size != coordinates.size:
            extra_keys = np.setdiff1d(abs(coordinates), abs(self.coordinate))
            if extra_keys.size == 0:
                raise ValueError(
                    'Duplicate coordinate values requested.  Please ensure coordinate indices are unique.')
            raise ValueError(
                'Not all indices in requested coordinate array exist in the system\n{:}'.format(str(extra_keys)))
        # Handle sign flipping
        multiplications = coordinates.flatten()[request_indices].sign(
        ) * self.coordinate[shape_indices].sign()
        return_value = self.transformation[shape_indices] * multiplications[:, np.newaxis]
        # Invert the indices to return the dofs in the correct order as specified in keys
        inverse_indices = np.zeros(request_indices.shape, dtype=int)
        inverse_indices[request_indices] = np.arange(len(request_indices))
        return_value = return_value[inverse_indices]
        return return_value

    def substructure_by_coordinate(self, dof_pairs, rcond=None,
                                   return_constrained_system=True):
        """
        Constrain the system by connecting the specified degree of freedom pairs

        Parameters
        ----------
        dof_pairs : iterable of CoordinateArray
            Pairs of coordinates to be connected.  None can be passed instead of
            a second degree of freedom to constrain to ground
        rcond : float, optional
            Condition threshold to use for the nullspace calculation on the
            constraint matrix. The default is None.
        return_constrained_system : bool, optional
            If true, apply the constraint matrix and return the constrained
            system, otherwise simply return the constraint matrix. The default
            is True.

        Returns
        -------
        np.ndarray or System
            Returns a System object with the constraints applied if
            `return_constrained_system` is True, otherwise just return the
            constraint matrix.

        """
        constraint_matrix = []
        for constraint_dof_0, constraint_dof_1 in dof_pairs:
            constraint = self.transformation_matrix_at_coordinates(constraint_dof_0)
            if constraint_dof_1 is not None:
                constraint -= self.transformation_matrix_at_coordinates(constraint_dof_1)
            constraint_matrix.append(constraint)
        constraint_matrix = np.concatenate(constraint_matrix, axis=0)
        if return_constrained_system:
            return self.constrain(constraint_matrix, rcond)
        else:
            return constraint_matrix

    def substructure_by_shape(self, constraint_shapes, connection_dofs_0,
                              connection_dofs_1=None, rcond=None,
                              return_constrained_system=True):
        """
        Constrain the system using a set of shapes in a least-squares sense.

        Parameters
        ----------
        constraint_shapes : ShapeArray
            An array of shapes to use as the basis for the constraints
        connection_dofs_0 : CoordinateArray
            Array of coordinates to use in the constraints
        connection_dofs_1 : CoordinateArray, optional
            Array of coordinates to constrain to the coordinates in
            `connection_dofs_0`. If not specified, the `connection_dofs_0`
            degrees of freedom will be constrained to ground.
        rcond : float, optional
            Condition threshold on the nullspace calculation. The default is None.
        return_constrained_system : bool, optional
            If true, apply the constraint matrix and return the constrained
            system, otherwise simply return the constraint matrix. The default
            is True.

        Returns
        -------
        np.ndarray or System
            Returns a System object with the constraints applied if
            `return_constrained_system` is True, otherwise just return the
            constraint matrix.

        """
        shape_matrix_0 = constraint_shapes[connection_dofs_0].T
        transform_matrix_0 = self.transformation_matrix_at_coordinates(connection_dofs_0)
        constraint_matrix = np.linalg.lstsq(shape_matrix_0, transform_matrix_0)[0]
        if connection_dofs_1 is not None:
            shape_matrix_1 = constraint_shapes[connection_dofs_1].T
            transform_matrix_1 = self.transformation_matrix_at_coordinates(connection_dofs_1)
            constraint_matrix -= np.linalg.lstsq(shape_matrix_1, transform_matrix_1)[0]
        if return_constrained_system:
            return self.constrain(constraint_matrix, rcond)
        else:
            return constraint_matrix

    def copy(self):
        """
        Returns a copy of the system object

        Returns
        -------
        System
            A copy of the system object

        """
        return System(self.coordinate.copy(), self.mass.copy(), self.stiffness.copy(),
                      self.damping.copy(), self.transformation.copy())

    def set_proportional_damping(self, mass_fraction, stiffness_fraction):
        """
        Sets the damping matrix to a proportion of the mass and stiffness matrices.

        The damping matrix will be set to `mass_fraction*self.mass +
        stiffness_fraction*self.stiffness`

        Parameters
        ----------
        mass_fraction : float
            Fraction of the mass matrix
        stiffness_fraction : TYPE
            Fraction of the stiffness matrix

        Returns
        -------
        None.

        """
        self.damping = self.mass * mass_fraction + self.stiffness * stiffness_fraction

    @classmethod
    def beam(cls, length, width, height, num_nodes, E=None, rho=None, nu=None, material=None):
        """
        Create a beam mass and stiffness matrix

        Parameters
        ----------
        length : float
            Lenghth of the beam
        width : float
            Width of the beam
        height : float
            Height of the beam
        num_nodes : int
            Number of nodes in the beam.
        E : float, optional
            Young's modulus of the beam. If not specified, a `material` must be
            specified instead
        rho : float, optional
            Density of the beam. If not specified, a `material` must be
            specified instead
        nu : float, optional
            Poisson's ratio of the beam. If not specified, a `material` must be
            specified instead
        material : str, optional
            A specific material can be specified instead of `E`, `rho`, and `nu`.
            Should be a string 'steel' or 'aluminum'.  If not specified, then
            options `E`, `rho`, and `nu` must be specified instead.

        Raises
        ------
        ValueError
            If improper materials are defined.

        Returns
        -------
        system : System
            A system object consisting of the beam mass and stiffness matrices.
        geometry : Geometry
            A Geometry consisting of the beam geometry.

        """
        from .sdynpy_geometry import Geometry, node_array, traceline_array, coordinate_system_array
        node_positions = np.array((np.linspace(0, length, num_nodes),
                                   np.zeros(num_nodes),
                                   np.zeros(num_nodes))).T
        node_connectivity = np.array((np.arange(num_nodes - 1), np.arange(1, num_nodes))).T
        bend_direction_1 = np.array((np.zeros(num_nodes - 1),
                                     np.zeros(num_nodes - 1),
                                     np.ones(num_nodes - 1))).T
        if material is None:
            if E is None or rho is None or nu is None:
                raise ValueError('Must specify material or E, nu, and rho')
        elif material.lower() == 'steel':
            E = 200e9  # [N/m^2],
            nu = 0.25  # [-],
            rho = 7850  # [kg/m^3]
        elif material.lower() == 'aluminum':
            E = 69e9  # [N/m^2],
            nu = 0.33  # [-],
            rho = 2830  # [kg/m^3]
        else:
            raise ValueError('Unknown Material {:}'.format(material))
        mat_props = rect_beam_props(E, rho, nu, width, height, num_nodes - 1)
        K, M = beamkm(node_positions, node_connectivity, bend_direction_1, **mat_props)
        coordinates = from_nodelist(np.arange(num_nodes) + 1, directions=[1, 2, 3, 4, 5, 6])
        system = cls(coordinates, M, K)
        nodelist = node_array(np.arange(num_nodes) + 1, node_positions)
        tracelines = traceline_array(connectivity=np.arange(num_nodes) + 1)
        coordinate_systems = coordinate_system_array()
        geometry = Geometry(nodelist, coordinate_systems, tracelines)
        return system, geometry

    def get_indices_by_coordinate(self, coordinates, ignore_sign=False):
        """
        Gets the indices in the transformation matrix corresponding coordinates

        Parameters
        ----------
        coordinates : CoordinateArray
            Coordinates to extract transformation indices
        ignore_sign : bool, optional
            Specify whether or not to ignore signs on the coordinates.  If True,
            then '101X+' would match '101X+' or '101X-'. The default is False.

        Raises
        ------
        ValueError
            Raised if duplicate coordinates or coordinates not in the system
            are requested

        Returns
        -------
        np.ndarray
            Array of indices.

        """
        if ignore_sign:
            consistent_arrays, shape_indices, request_indices = np.intersect1d(
                abs(self.coordinate), abs(coordinates), assume_unique=False, return_indices=True)
        else:
            consistent_arrays, shape_indices, request_indices = np.intersect1d(
                self.coordinate, coordinates, assume_unique=False, return_indices=True)
        # Make sure that all of the keys are actually in the consistent array matrix
        if consistent_arrays.size != coordinates.size:
            extra_keys = np.setdiff1d(abs(coordinates), abs(self.coordinate))
            if extra_keys.size == 0:
                raise ValueError(
                    'Duplicate coordinate values requested.  Please ensure coordinate indices are unique.')
            raise ValueError(
                'Not all indices in requested coordinate array exist in the system\n{:}'.format(str(extra_keys)))
        # Handle sign flipping
        return_value = shape_indices
        # Invert the indices to return the dofs in the correct order as specified in keys
        inverse_indices = np.zeros(request_indices.shape, dtype=int)
        inverse_indices[request_indices] = np.arange(len(request_indices))
        return return_value[inverse_indices]

    def reduce(self, reduction_transformation):
        """
        Apply the specified reduction to the model

        Parameters
        ----------
        reduction_transformation : np.ndarray
            Matrix to use in the reduction

        Returns
        -------
        System
            Reduced system.

        """
        mass = reduction_transformation.T @ self.mass @ reduction_transformation
        stiffness = reduction_transformation.T @ self.stiffness @ reduction_transformation
        damping = reduction_transformation.T @ self.damping @ reduction_transformation
        transformation = self.transformation @ reduction_transformation
        # Force symmetry
        mass = (mass + mass.T) / 2
        stiffness = (stiffness + stiffness.T) / 2
        damping = (damping + damping.T) / 2
        return System(self.coordinate, mass, stiffness, damping, transformation)

    def reduce_guyan(self, coordinates):
        """
        Perform Guyan reduction on the system

        Parameters
        ----------
        coordinates : CoordinateArray
            A list of coordinates to keep in the reduced system.

        Raises
        ------
        ValueError
            Raised the transformation matrix is not identity matrix.

        Returns
        -------
        System
            Reduced system.

        """
        if isinstance(coordinates, CoordinateArray):
            if not np.allclose(self.transformation, np.eye(*self.transformation.shape)):
                raise ValueError(
                    'Coordinates can only be specified with a CoordinateArray if the transformation is identity')
            keep_dofs = self.get_indices_by_coordinate(coordinates)
        else:
            keep_dofs = np.array(coordinates)
        discard_dofs = np.array([i for i in range(self.ndof) if i not in keep_dofs])
        I_a = np.eye(keep_dofs.size)
        K_dd = self.stiffness[discard_dofs[:, np.newaxis],
                              discard_dofs]
        K_da = self.stiffness[discard_dofs[:, np.newaxis],
                              keep_dofs]
        T_guyan = np.concatenate((I_a, -np.linalg.solve(K_dd, K_da)), axis=0)
        T_guyan[np.concatenate((keep_dofs, discard_dofs)), :] = T_guyan.copy()
        return self.reduce(T_guyan)

    def reduce_dynamic(self, coordinates, frequency):
        """
        Perform Dynamic condensation

        Parameters
        ----------
        coordinates : CoordinateArray
            A list of coordinates to keep in the reduced system.
        frequency : float
            The frequency to preserve in the dynamic reduction.

        Raises
        ------
        ValueError
            Raised if the transformation is not identity matrix.

        Returns
        -------
        System
            Reduced system.

        """
        if isinstance(coordinates, CoordinateArray):
            if not np.allclose(self.transformation, np.eye(*self.transformation.shape)):
                raise ValueError(
                    'Coordinates can only be specified with a CoordinateArray if the transformation is identity')
            keep_dofs = self.get_indices_by_coordinate(coordinates)
        else:
            keep_dofs = np.array(coordinates)
        discard_dofs = np.array([i for i in range(self.ndof) if i not in keep_dofs])
        I_a = np.eye(keep_dofs.size)
        D = self.stiffness - (2 * np.pi * frequency)**2 * self.mass
        D_dd = D[discard_dofs[:, np.newaxis],
                 discard_dofs]
        D_da = D[discard_dofs[:, np.newaxis],
                 keep_dofs]
        T_dynamic = np.concatenate((I_a, -np.linalg.solve(D_dd, D_da)), axis=0)
        T_dynamic[np.concatenate((keep_dofs, discard_dofs)), :] = T_dynamic.copy()
        return self.reduce(T_dynamic)

    def reduce_craig_bampton(self, connection_degrees_of_freedom: CoordinateArray,
                             num_fixed_base_modes: int,
                             return_shape_matrix: bool = False):
        """
        Computes a craig-bampton substructure model for the system

        Parameters
        ----------
        connection_degrees_of_freedom : CoordinateArray
            Degrees of freedom to keep at the interface.
        num_fixed_base_modes : int
            Number of fixed-base modes to use in the reduction
        return_shape_matrix : bool, optional
            If true, return a set of shapes that represents the transformation
            in addition to the reduced system.  The default is False.

        Raises
        ------
        ValueError
            Raised if coordinate arrays are specified when there is already a
            transformation.

        Returns
        -------
        System
            Reduced system in craig-bampton form
        ShapeArray
            Shapes representing the craig-bampton transformation

        """
        # Construct craig bampton transformation
        if isinstance(connection_degrees_of_freedom, CoordinateArray):
            if not np.allclose(self.transformation, np.eye(*self.transformation.shape)):
                raise ValueError(
                    'Coordinates can only be specified with a CoordinateArray if the transformation is identity')
            connection_indices = self.get_indices_by_coordinate(connection_degrees_of_freedom)
        else:
            connection_indices = np.array(connection_degrees_of_freedom)
        other_indices = np.array([i for i in range(self.ndof) if i not in connection_indices])

        # Extract portions of the mass and stiffness matrices
        K_ii = self.K[other_indices[:, np.newaxis], other_indices]
        M_ii = self.M[other_indices[:, np.newaxis], other_indices]
        K_ib = self.K[other_indices[:, np.newaxis], connection_indices]

        # Compute fixed interface modes
        lam, Phi_ii = eigh(K_ii, M_ii, subset_by_index=[0, int(num_fixed_base_modes) - 1])
        # Normalize the mode shapes
        lam[lam < 0] = 0
        normalized_mass = np.diag(Phi_ii.T @ M_ii @ Phi_ii)
        Phi_ii /= np.sqrt(normalized_mass)
        Z_bi = np.zeros((connection_indices.size, num_fixed_base_modes))
        # Compute constraint modes
        Psi_ib = -np.linalg.solve(K_ii, K_ib)
        I_bb = np.eye(connection_indices.size)
        T_cb = np.block([[Phi_ii, Psi_ib],
                         [Z_bi, I_bb]])
        T_cb[np.concatenate((other_indices, connection_indices)), :] = T_cb.copy()
        if return_shape_matrix:
            from .sdynpy_shape import shape_array
            freq = np.sqrt(lam) / (2 * np.pi)
            all_freqs = np.concatenate((freq, np.zeros(connection_indices.size)))
            shapes = shape_array(self.coordinate, T_cb.T, all_freqs, comment1=['Fixed Base Mode {:}'.format(
                i + 1) for i in range(num_fixed_base_modes)] + ['Constraint Mode {:}'.format(str(dof)) for dof in connection_degrees_of_freedom])
            return self.reduce(T_cb), shapes
        else:
            return self.reduce(T_cb)

    @classmethod
    def from_exodus_superelement(cls, superelement_nc4, transformation_exodus_file=None,
                                 x_disp='DispX', y_disp='DispY', z_disp='DispZ',
                                 x_rot=None, y_rot=None, z_rot=None,
                                 reduce_to_external_surfaces=False):
        """
        Creates a system from a superelement from Sierra/SD

        Parameters
        ----------
        superelement_nc4 : netCDF4.Dataset or string
            Dataset from which the superelement data will be loaded
        transformation_exodus_file : Exodus, ExodusInMemory, or str, optional
            Exodus data containing the transformation between the reduced
            superelement state and the physical space.  If not specified, no
            transformation will be created.
        x_disp : str, optional
            Variable name to read for x-displacements in the transformation
            Exodus file. The default is 'DispX'.
        y_disp : str, optional
            Variable name to read for y-displacements in the transformation
            Exodus file. The default is 'DispY'.
        z_disp : str, optional
            Variable name to read for z-displacements in the transformation
            Exodus file. The default is 'DispZ'.
        x_rot : str, optional
            Variable name to read for x-rotations in the transformation
            Exodus file. The default is to not read rotations.
        y_rot : str, optional
            Variable name to read for y-rotations in the transformation
            Exodus file. The default is to not read rotations.
        z_rot : str, optional
            Variable name to read for z-rotations in the transformation
            Exodus file. The default is to not read rotations.
        reduce_to_external_surfaces : bool, optional
            If True, exodus results will be reduced to external surfaces

        Raises
        ------
        ValueError
            raised if bad data types are passed to the arguments.

        Returns
        -------
        system : System
            System containing the superelement representation
        geometry : Geometry
            Geometry that can be used to plot the system
        boundary_dofs : CoordinateArray
            Degrees of freedom that can be used to constrain the test article.

        """
        from .sdynpy_geometry import node_array, coordinate_system_array, Geometry
        if isinstance(superelement_nc4, str):
            ds = nc4.Dataset(superelement_nc4)
        elif isinstance(superelement_nc4, nc4.Dataset):
            ds = superelement_nc4
        else:
            raise ValueError('superelement_nc4 must be a string or a netCDF4 Dataset')
        cbmap = ds['cbmap'][:].data.copy()
        Kr = ds['Kr'][:].data.copy()
        Mr = ds['Mr'][:].data.copy()
        Cr = ds['Cr'][:].data.copy()
        num_constraint_modes = ds.dimensions['NumConstraints'].size
        num_fixed_base_modes = ds.dimensions['NumEig'].size
        boundary_dofs = coordinate_array(*cbmap[cbmap[:, 0] > 0].T)
        if transformation_exodus_file is None:
            transformation = None
            coordinate_nodes = cbmap[:, 0]
            coordinate_dirs = cbmap[:, 1]
            coordinate_nodes[coordinate_nodes == 0] = np.arange(num_fixed_base_modes) + 1
            coordinates = coordinate_array(coordinate_nodes, coordinate_dirs)
            cs_array = coordinate_system_array()
            n_array = node_array(ds['node_num_map'][:], np.array(
                [ds['coord{:}'.format(d)][:] for d in 'xyz']).T)
            geometry = Geometry(node=n_array, coordinate_system=cs_array)
        else:
            if isinstance(transformation_exodus_file, str):
                exo = Exodus(transformation_exodus_file)
            elif isinstance(transformation_exodus_file, Exodus):
                exo = transformation_exodus_file
            elif isinstance(transformation_exodus_file, ExodusInMemory):
                exo = transformation_exodus_file
            else:
                raise ValueError('transformation_exodus_file must be a string or a sdpy.Exodus')
            if reduce_to_external_surfaces:
                exo = reduce_exodus_to_surfaces(exo, variables_to_transform=[var for var in [x_disp, y_disp, z_disp, x_rot, y_rot, z_rot] if var is not None])
            from .sdynpy_shape import ShapeArray
            shapes = ShapeArray.from_exodus(exo, x_disp, y_disp, z_disp, x_rot, y_rot, z_rot)
            transformation = shapes.shape_matrix.T
            coordinates = shapes[0].coordinate
            geometry = Geometry.from_exodus(exo)
        system = cls(coordinates, Mr, Kr, Cr, transformation)
        return system, geometry, boundary_dofs

    def simulate_test(
            self,  # The system itself
            bandwidth,
            frame_length,
            num_averages,
            excitation,
            references,
            responses=None,  # All Responses
            excitation_level=1.0,
            excitation_noise_level=0.0,
            response_noise_level=0.0,
            steady_state_time=0.0,
            excitation_min_frequency=None,
            excitation_max_frequency=None,
            signal_fraction=0.5,
            extra_time_between_frames=0.0,
            integration_oversample=10,
            displacement_derivative=2,
            antialias_filter_cutoff_factor=3,
            antialias_filter_order=4,
            **generator_kwargs
    ):
        available_excitations = ['pseudorandom', 'random',
                                 'burst random', 'chirp', 'hammer', 'multi-hammer', 'sine']
        if not excitation.lower() in available_excitations:
            raise ValueError('Excitation must be one of {:}'.format(available_excitations))
        # Create the input signal
        num_signals = references.size
        sample_rate = bandwidth * 2 * integration_oversample
        dt = 1 / sample_rate
        frame_time = dt * frame_length * integration_oversample
        df = 1 / frame_time
        # Create the signals
        if excitation.lower() == 'pseudorandom':
            if num_signals > 1:
                print('Warning: Pseudorandom generally not recommended for multi-reference excitation.')
            kwargs = {'fft_lines': frame_length // 2,
                      'f_nyq': bandwidth,
                      'signal_rms': excitation_level,
                      'min_freq': excitation_min_frequency,
                      'max_freq': excitation_max_frequency,
                      'integration_oversample': integration_oversample,
                      'averages': num_averages + int(np.ceil(steady_state_time / frame_time))}
            kwargs.update(generator_kwargs)
            signals = np.array([
                generator.pseudorandom(**kwargs)[1]
                for i in range(num_signals)
            ])
        elif excitation.lower() == 'random':
            kwargs = {'shape': (num_signals,),
                      'n_samples': frame_length * integration_oversample * num_averages + int(steady_state_time * sample_rate),
                      'rms': excitation_level,
                      'dt': dt,
                      'low_frequency_cutoff': excitation_min_frequency,
                      'high_frequency_cutoff': bandwidth if excitation_max_frequency is None else excitation_max_frequency}
            kwargs.update(generator_kwargs)
            signals = generator.random(**kwargs)
        elif excitation.lower() == 'burst random':
            kwargs = {'shape': (num_signals,),
                      'n_samples': frame_length * integration_oversample,
                      'on_fraction': signal_fraction,
                      'delay_fraction': 0,
                      'rms': excitation_level,
                      'dt': dt,
                      'low_frequency_cutoff': excitation_min_frequency,
                      'high_frequency_cutoff': bandwidth if excitation_max_frequency is None else excitation_max_frequency}
            kwargs.update(generator_kwargs)
            signal_list = [generator.burst_random(**kwargs) for i in range(num_averages)]
            full_list = []
            for i, signal in enumerate(signal_list):
                full_list.append(
                    np.zeros((num_signals, int(extra_time_between_frames * sample_rate))))
                full_list.append(signal)
            signals = np.concatenate(full_list, axis=-1)
        elif excitation.lower() == 'chirp':
            if num_signals > 1:
                print('Warning: Chirp generally not recommended for multi-reference excitation.')
            kwargs = {'frequency_min': 0 if excitation_min_frequency is None else excitation_min_frequency,
                      'frequency_max': bandwidth if excitation_max_frequency is None else excitation_max_frequency,
                      'signal_length': frame_time,
                      'dt': dt}
            kwargs.update(generator_kwargs)
            signals = np.array([
                generator.chirp(**kwargs)
                for i in range(num_signals)
            ]) * excitation_level
            signals = np.tile(signals, [1, num_averages +
                              int(np.ceil(steady_state_time / frame_time))])
        elif excitation.lower() == 'hammer':
            if num_signals > 1:
                print(
                    'Warning: Hammer impact generally not recommended for multi-reference excitation, consider multi-hammer instead')
            pulse_width = 2 / (bandwidth if excitation_max_frequency is None else excitation_max_frequency)
            signal_length = int(frame_length * integration_oversample * num_averages + (num_averages + 1)
                                * extra_time_between_frames * sample_rate + 2 * pulse_width * sample_rate)
            pulse_times = np.arange(num_averages)[
                :, np.newaxis] * (frame_time + extra_time_between_frames) + pulse_width + extra_time_between_frames
            kwargs = {'signal_length': signal_length,
                      'pulse_time': pulse_times,
                      'pulse_width': pulse_width,
                      'pulse_peak': excitation_level,
                      'dt': dt,
                      'sine_exponent': 2}
            kwargs.update(generator_kwargs)
            signals = generator.pulse(**kwargs)
            signals = np.tile(signals, [num_signals, 1])
        elif excitation.lower() == 'multi-hammer':
            signal_length = frame_length * integration_oversample
            pulse_width = 2 / (bandwidth if excitation_max_frequency is None else excitation_max_frequency)
            signals = []
            for i in range(num_signals):
                signals.append([])
                for j in range(num_averages):
                    pulse_times = []
                    last_pulse = 0
                    while last_pulse < frame_time * signal_fraction:
                        next_pulse = last_pulse + pulse_width * (np.random.rand() * 4 + 1)
                        pulse_times.append(next_pulse)
                        last_pulse = next_pulse
                    pulse_times = np.array(pulse_times)
                    pulse_times = pulse_times[pulse_times <
                                              frame_time * signal_fraction, np.newaxis]
                    kwargs = {'signal_length': signal_length,
                              'pulse_time': pulse_times,
                              'pulse_width': pulse_width,
                              'pulse_peak': excitation_level,
                              'dt': dt,
                              'sine_exponent': 2}
                    kwargs.update(generator_kwargs)
                    signal = generator.pulse(**kwargs)
                    signals[-1].append(np.zeros(int(extra_time_between_frames * sample_rate)))
                    signals[-1].append(signal)
                signals[-1].append(np.zeros(int(extra_time_between_frames * sample_rate)))
                signals[-1] = np.concatenate(signals[-1], axis=-1)
            signals = np.array(signals)
        elif excitation.lower() == 'sine':
            if num_signals > 1:
                print(
                    'Warning: Sine signal generally not recommended for multi-reference excitation')
            frequencies = excitation_max_frequency if excitation_min_frequency is None else excitation_min_frequency
            num_samples = frame_length * integration_oversample * num_averages + int(steady_state_time * sample_rate)
            kwargs = {'frequencies': frequencies,
                      'dt': dt,
                      'num_samples': num_samples,
                      'amplitudes': excitation_level}
            kwargs.update(generator_kwargs)
            signals = np.tile(generator.sine(**kwargs), (num_signals, 1))
        # Set up the integration
        responses, references = self.time_integrate(
            signals, dt, responses, references, displacement_derivative)
        # Now add noise
        responses.ordinate += response_noise_level * np.random.randn(*responses.ordinate.shape)
        references.ordinate += excitation_noise_level * np.random.randn(*references.ordinate.shape)
        # Filter with antialiasing filters, divide filter order by 2 because of filtfilt
        if antialias_filter_order > 0:
            lowpass_b, lowpass_a = butter(antialias_filter_order // 2,
                                          antialias_filter_cutoff_factor * bandwidth, fs=sample_rate)
            responses.ordinate = filtfilt(lowpass_b, lowpass_a, responses.ordinate)
            references.ordinate = filtfilt(lowpass_b, lowpass_a, references.ordinate)
        if integration_oversample > 1:
            responses = responses.downsample(integration_oversample)
            references = references.downsample(integration_oversample)
        responses = responses.extract_elements_by_abscissa(steady_state_time, np.inf)
        references = references.extract_elements_by_abscissa(steady_state_time, np.inf)
        return responses, references

    # def reduce_serep(self,shapes_full,shapes_reduced):
    #     if isinstance(shapes_full,)


substructure_by_position = System.substructure_by_position


class SystemSparse(System):
    """
    Matrix Equations representing a Structural Dynamics System using sparse matrices
    
    This class extends the System class to handle large FEM models using scipy sparse
    CSR matrices for mass, stiffness, damping, and transformation matrices.
    """

    def __init__(self, coordinate: CoordinateArray, mass, stiffness, damping=None,
                 transformation=None):
        """
        Create a sparse system representation including mass, stiffness, damping, and
        transformation matrices.

        Parameters
        ----------
        coordinate : CoordinateArray
            Physical degrees of freedom in the system.
        mass : scipy.sparse matrix
            Sparse matrix consisting of the mass matrix of the system
        stiffness : scipy.sparse matrix
            Sparse matrix consisting of the stiffness matrix of the system
        damping : scipy.sparse matrix, optional
            Sparse matrix consisting of the damping matrix of the system.  If not
            specified, the damping will be zero.
        transformation : scipy.sparse matrix, optional
            A transformation between internal "state" degrees of freedom and
            the physical degrees of freedom defined in `coordinate`. The
            default transformation is the identity matrix.

        Raises
        ------
        ValueError
            If inputs are improperly sized

        Returns
        -------
        None.
        """
        # Convert to CSR format if not already
        mass = sp.csr_matrix(mass)
        stiffness = sp.csr_matrix(stiffness)
        
        if not mass.shape == stiffness.shape:
            raise ValueError('Mass and Stiffness matrices must be the same shape')
        if not mass.ndim == 2 or (mass.shape[0] != mass.shape[1]):
            raise ValueError('Mass should be a 2D, square array')
        if not stiffness.ndim == 2 or (stiffness.shape[0] != stiffness.shape[1]):
            raise ValueError('Stiffness should be a 2D, square array')
        
        if damping is None:
            damping = sp.csr_matrix((stiffness.shape))
        else:
            damping = sp.csr_matrix(damping)
        if not damping.shape == stiffness.shape:
            raise ValueError('Damping and Stiffness matrices must be the same shape')
        if not damping.ndim == 2 or (damping.shape[0] != damping.shape[1]):
            raise ValueError('Damping should be a 2D, square array')
        
        if transformation is None:
            transformation = sp.identity(mass.shape[0], format='csr')
        else:
            transformation = sp.csr_matrix(transformation)
        if not transformation.ndim == 2:
            raise ValueError('transformation must be 2D')
        if not transformation.shape[-1] == mass.shape[0]:
            raise ValueError(
                'transformation must have number of columns equal to the number of rows in the mass matrix')
        
        coordinate = np.atleast_1d(coordinate)
        if not isinstance(coordinate, CoordinateArray):
            raise ValueError('coordinate must be a CoordinateArray object')
        if not coordinate.ndim == 1 or coordinate.shape[0] != transformation.shape[0]:
            raise ValueError(
                'coordinate must be 1D and have the same size as transformation.shape[0] or mass.shape[0] if no transformation is specified')
        
        # Check symmetry on a subset of non-zero elements for efficiency
        self._check_sparse_symmetry(mass, 'mass matrix')
        self._check_sparse_symmetry(stiffness, 'stiffness matrix')
        self._check_sparse_symmetry(damping, 'damping matrix')

        self._coordinate = coordinate
        self._mass = mass
        self._stiffness = stiffness
        self._damping = damping
        self._transformation = transformation

    def _check_sparse_symmetry(self, matrix, name, sample_size=1000):
        """
        Check symmetry of sparse matrix by sampling non-zero elements
        
        Parameters
        ----------
        matrix : scipy.sparse matrix
            Matrix to check for symmetry
        name : str
            Name of matrix for error messages
        sample_size : int, optional
            Number of non-zero elements to sample for symmetry check
        """
        if matrix.nnz == 0:
            return  # Empty matrix is symmetric
            
        # Get non-zero elements
        rows, cols = matrix.nonzero()
        data = matrix.data
        
        # Sample a subset if matrix is large
        if len(rows) > sample_size:
            indices = np.random.choice(len(rows), sample_size, replace=False)
            rows = rows[indices]
            cols = cols[indices]
            data = data[indices]
        
        # Check symmetry for sampled elements
        for i, (r, c) in enumerate(zip(rows, cols)):
            if r != c:  # Only check off-diagonal elements
                val_rc = data[i]
                val_cr = matrix[c, r]
                if not np.allclose(val_rc, val_cr, atol=1e-6*np.abs(val_rc).max() if np.abs(val_rc).max() > 0 else 1e-6):
                    raise ValueError(f'{name} must be symmetric')

    @property
    def mass(self):
        """Get or set the mass matrix of the system"""
        return self._mass

    @mass.setter
    def mass(self, value):
        """Set the mass matrix of the system"""
        value = sp.csr_matrix(value)
        if not value.shape == self.stiffness.shape:
            raise ValueError('Mass and Stiffness matrices must be the same shape')
        if not value.ndim == 2 or (value.shape[0] != value.shape[1]):
            raise ValueError('Mass should be a 2D, square array')
        self._check_sparse_symmetry(value, 'mass matrix')
        self._mass = value

    @property
    def stiffness(self):
        """Get or set the stiffness matrix of the system"""
        return self._stiffness

    @stiffness.setter
    def stiffness(self, value):
        """Set the stiffness matrix of the system"""
        value = sp.csr_matrix(value)
        if not value.shape == self.mass.shape:
            raise ValueError('Mass and Stiffness matrices must be the same shape')
        if not value.ndim == 2 or (value.shape[0] != value.shape[1]):
            raise ValueError('Stiffness should be a 2D, square array')
        self._check_sparse_symmetry(value, 'stiffness matrix')
        self._stiffness = value

    @property
    def damping(self):
        """Get or set the damping matrix of the system"""
        return self._damping

    @damping.setter
    def damping(self, value):
        """Set the damping matrix of the system"""
        value = sp.csr_matrix(value)
        if not value.shape == self.stiffness.shape:
            raise ValueError('damping and Stiffness matrices must be the same shape')
        if not value.ndim == 2 or (value.shape[0] != value.shape[1]):
            raise ValueError('damping should be a 2D, square array')
        self._check_sparse_symmetry(value, 'damping matrix')
        self._damping = value

    @property
    def transformation(self):
        """Get or set the transformation matrix"""
        return self._transformation

    @transformation.setter
    def transformation(self, value):
        """Set the transformation matrix"""
        value = sp.csr_matrix(value)
        if not value.ndim == 2:
            raise ValueError('transformation must be 2D')
        if not value.shape[-1] == self.mass.shape[0]:
            raise ValueError(
                'transformation must have number of columns equal to the number of rows in the mass matrix')
        self._transformation = value

    # Alias properties for convenience
    M = mass
    K = stiffness
    C = damping

    def spy(self, subplots_kwargs={'figsize': (10, 3)}, spy_kwargs={}):
        """
        Plot the structure of the sparse system's matrices

        Parameters
        ----------
        subplots_kwargs : dict, optional
            Default arguments passed to `matplotlib.pyplot`'s `subplots` function.
            The default is {'figsize':(10,3)}.
        spy_kwargs : dict, optional
            Default arguments passed to `matplotlib.pyplot`'s 'spy' function.
            The default is {}.

        Returns
        -------
        ax : Axes
            Axes on which the subplots are defined.
        """
        fig, ax = plt.subplots(1, 3, squeeze=True, **subplots_kwargs)
        
        # Use sparse-specific spy plotting
        ax[0].spy(self.transformation, **spy_kwargs)
        
        # Combine system matrices for visualization
        combined = abs(self.mass) + abs(self.stiffness) + abs(self.damping)
        ax[1].spy(combined, **spy_kwargs)
        
        ax[2].spy(self.transformation.T, **spy_kwargs)
        
        ax[0].set_title('Output Transformation')
        ax[1].set_title('Internal State Matrices')
        ax[2].set_title('Input Transformation')
        fig.tight_layout()
        return ax

    def eigensolution(self, num_modes=None, maximum_frequency=None, complex_modes=False, return_shape=True):
        """
        Computes the eigensolution of the sparse system

        Parameters
        ----------
        num_modes : int, optional
            The number of modes of the system to compute. The default is to
            compute all the modes.
        maximum_frequency : float, optional
            The maximum frequency to which modes will be computed.
            The default is to compute all the modes.
        complex_modes : bool, optional
            Whether or not complex modes are computed. The default is False.
        return_shape : bool, optional
            Specifies whether or not to return a `ShapeArray` (True) or a reduced
            `System` (False). The default is True.

        Returns
        -------
        System or ShapeArray
            If `return_shape` is True, then a ShapeArray will be returned.  If
            `return_shape` is False, a reduced System will be returned.
        """
        if complex_modes:
            raise NotImplementedError('Complex modes not yet implemented for sparse systems')
        
        # Use sparse eigenvalue solver
        if num_modes is None:
            num_modes = min(self.ndof - 2, 50)  # Default to reasonable number for sparse
        
        # Ensure we don't ask for more modes than possible
        max_modes = min(num_modes, self.ndof - 2)
        
        # Convert frequency to eigenvalue range if specified
        if maximum_frequency is not None:
            sigma = (2 * np.pi * maximum_frequency)**2
            # Use shift-invert mode to find eigenvalues near sigma
            lam, phi = spla.eigsh(self.K, max_modes, self.M, sigma=sigma, which='LM', tol=1e-3)
        else:
            # Find smallest eigenvalues using shift-invert
            sigma = 1.0
            lam, phi = spla.eigsh(self.K, max_modes, self.M, sigma=sigma, which='LM', tol=1e-3)
        
        # Sort by frequency
        idx = np.argsort(lam)
        lam = lam[idx]
        phi = phi[:, idx]
        
        # Mass normalize the mode shapes
        lam[lam < 0] = 0
        freq = np.sqrt(lam) / (2 * np.pi)
        
        # Convert to dense for mass normalization
        phi_dense = phi
        mass_dense = self.M.toarray()
        damping_dense = self.C.toarray()
        
        normalized_mass = np.diag(phi_dense.T @ mass_dense @ phi_dense)
        phi_dense /= np.sqrt(normalized_mass)
        
        # Compute modal damping
        with np.errstate(divide='ignore', invalid='ignore'):
            damping = np.diag(phi_dense.T @ damping_dense @ phi_dense) / (2 * (2 * np.pi * freq))
        damping[np.isnan(damping)] = 0.0
        damping[np.isinf(damping)] = 0.0
        
        # Apply transformation to get back to physical dofs
        phi_physical = self.transformation @ phi_dense
        
        if return_shape:
            from .sdynpy_shape import shape_array
            return shape_array(self.coordinate, phi_physical.T, freq, damping)
        else:
            # Return a dense System for the modal system
            return System(self.coordinate, 
                         np.eye(freq.size), 
                         np.diag((2 * np.pi * freq)**2), 
                         np.diag(2 * (2 * np.pi * freq) * damping), 
                         phi_physical)

    def reduce(self, reduction_transformation):
        """
        Apply the specified reduction to the sparse model

        Parameters
        ----------
        reduction_transformation : np.ndarray or scipy.sparse matrix
            Matrix to use in the reduction

        Returns
        -------
        System
            Reduced system as a dense System object.
        """
        if not sp.issparse(reduction_transformation):
            reduction_transformation = sp.csr_matrix(reduction_transformation)
        
        # Perform sparse matrix operations
        mass = reduction_transformation.T @ self.mass @ reduction_transformation
        stiffness = reduction_transformation.T @ self.stiffness @ reduction_transformation
        damping = reduction_transformation.T @ self.damping @ reduction_transformation
        transformation = self.transformation @ reduction_transformation
        
        # Convert to dense arrays for the reduced system
        mass_dense = mass.toarray()
        stiffness_dense = stiffness.toarray()
        damping_dense = damping.toarray()
        transformation_dense = transformation.toarray()
        
        # Force symmetry
        mass_dense = (mass_dense + mass_dense.T) / 2
        stiffness_dense = (stiffness_dense + stiffness_dense.T) / 2
        damping_dense = (damping_dense + damping_dense.T) / 2
        
        return System(self.coordinate, mass_dense, stiffness_dense, damping_dense, transformation_dense)

    def constrain(self, constraint_matrix, rcond=None):
        """
        Apply a constraint matrix to the sparse system

        Parameters
        ----------
        constraint_matrix : np.ndarray or scipy.sparse matrix
            A matrix of constraints to apply to the structure
        rcond : float, optional
            Condition tolerance for computing the nullspace. The default is None.

        Returns
        -------
        System
            Constrained system as a dense System object.
        """
        if not sp.issparse(constraint_matrix):
            constraint_matrix = sp.csr_matrix(constraint_matrix)
        
        # Compute nullspace using dense conversion for now
        # TODO: Implement sparse nullspace computation
        constraint_dense = constraint_matrix.toarray()
        substructuring_transform_matrix = null_space(constraint_dense, rcond)
        
        return self.reduce(substructuring_transform_matrix)

    def transformation_matrix_at_coordinates(self, coordinates):
        """
        Return the transformation matrix at the specified coordinates

        Parameters
        ----------
        coordinates : CoordinateArray
            coordinates at which the transformation matrix will be computed.

        Returns
        -------
        return_value : scipy.sparse matrix
            Portion of the transformation matrix corresponding to the
            coordinates input to the function.
        """
        consistent_arrays, shape_indices, request_indices = np.intersect1d(
            abs(self.coordinate), abs(coordinates), assume_unique=False, return_indices=True)
        
        # Make sure that all of the keys are actually in the consistent array matrix
        if consistent_arrays.size != coordinates.size:
            extra_keys = np.setdiff1d(abs(coordinates), abs(self.coordinate))
            if extra_keys.size == 0:
                raise ValueError(
                    'Duplicate coordinate values requested.  Please ensure coordinate indices are unique.')
            raise ValueError(
                'Not all indices in requested coordinate array exist in the system\n{:}'.format(str(extra_keys)))
        
        # Handle sign flipping
        multiplications = coordinates.flatten()[request_indices].sign() * self.coordinate[shape_indices].sign()
        
        # Extract rows and apply sign corrections
        return_value = self.transformation[shape_indices]
        # Apply sign corrections element-wise to each row
        for i, mult in enumerate(multiplications):
            if mult != 1.0:
                return_value[i] = return_value[i] * mult
        
        # Invert the indices to return the dofs in the correct order as specified in keys
        inverse_indices = np.zeros(request_indices.shape, dtype=int)
        inverse_indices[request_indices] = np.arange(len(request_indices))
        return_value = return_value[inverse_indices]
        
        return return_value

    def copy(self):
        """
        Returns a copy of the sparse system object

        Returns
        -------
        SystemSparse
            A copy of the sparse system object
        """
        return SystemSparse(self.coordinate.copy(), 
                           self.mass.copy(), 
                           self.stiffness.copy(),
                           self.damping.copy(), 
                           self.transformation.copy())

    def set_proportional_damping(self, mass_fraction, stiffness_fraction):
        """
        Sets the damping matrix to a proportion of the mass and stiffness matrices.

        The damping matrix will be set to `mass_fraction*self.mass +
        stiffness_fraction*self.stiffness`

        Parameters
        ----------
        mass_fraction : float
            Fraction of the mass matrix
        stiffness_fraction : float
            Fraction of the stiffness matrix
        """
        self.damping = self.mass * mass_fraction + self.stiffness * stiffness_fraction

    def assign_modal_damping(self, damping_ratios):
        """
        Assigns a damping matrix to the sparse system that results in equivalent
        modal damping

        Parameters
        ----------
        damping_ratios : ndarray
            An array of damping values to assign to the system
        """
        damping_ratios = np.array(damping_ratios)
        if damping_ratios.ndim == 1:
            shapes = self.eigensolution(num_modes=damping_ratios.size)
        else:
            shapes = self.eigensolution()
        shapes.damping = damping_ratios
        # Compute the damping matrix
        modal_system = shapes.system()
        shape_pinv = np.linalg.pinv(modal_system.transformation.T)
        full_damping_matrix = shape_pinv @ modal_system.damping @ shape_pinv.T
        self.damping = sp.csr_matrix(full_damping_matrix)

    def get_indices_by_coordinate(self, coordinates, ignore_sign=False):
        """
        Gets the indices in the transformation matrix corresponding coordinates

        Parameters
        ----------
        coordinates : CoordinateArray
            Coordinates to extract transformation indices
        ignore_sign : bool, optional
            Specify whether or not to ignore signs on the coordinates.  If True,
            then '101X+' would match '101X+' or '101X-'. The default is False.

        Returns
        -------
        np.ndarray
            Array of indices.
        """
        if ignore_sign:
            consistent_arrays, shape_indices, request_indices = np.intersect1d(
                abs(self.coordinate), abs(coordinates), assume_unique=False, return_indices=True)
        else:
            consistent_arrays, shape_indices, request_indices = np.intersect1d(
                self.coordinate, coordinates, assume_unique=False, return_indices=True)
        # Make sure that all of the keys are actually in the consistent array matrix
        if consistent_arrays.size != coordinates.size:
            extra_keys = np.setdiff1d(abs(coordinates), abs(self.coordinate))
            if extra_keys.size == 0:
                raise ValueError(
                    'Duplicate coordinate values requested.  Please ensure coordinate indices are unique.')
            raise ValueError(
                'Not all indices in requested coordinate array exist in the system\n{:}'.format(str(extra_keys)))
        # Handle sign flipping
        return_value = shape_indices
        # Invert the indices to return the dofs in the correct order as specified in keys
        inverse_indices = np.zeros(request_indices.shape, dtype=int)
        inverse_indices[request_indices] = np.arange(len(request_indices))
        return return_value[inverse_indices]

    def reduce_guyan(self, coordinates):
        """
        Perform Guyan reduction on the sparse system

        Parameters
        ----------
        coordinates : CoordinateArray
            A list of coordinates to keep in the reduced system.

        Returns
        -------
        System
            Reduced system as a dense System object.
        """
        if isinstance(coordinates, CoordinateArray):
            if not sp.issparse(self.transformation) or not np.allclose(self.transformation.toarray(), np.eye(*self.transformation.shape)):
                raise ValueError(
                    'Coordinates can only be specified with a CoordinateArray if the transformation is identity')
            keep_dofs = self.get_indices_by_coordinate(coordinates)
        else:
            keep_dofs = np.array(coordinates)
        discard_dofs = np.array([i for i in range(self.ndof) if i not in keep_dofs])
        I_a = sp.identity(keep_dofs.size, format='csr')
        K_dd = self.stiffness[discard_dofs][:, discard_dofs]
        K_da = self.stiffness[discard_dofs][:, keep_dofs]
        
        # Add small regularization to handle potential singularity from rigid body modes
        # Use setdiag to avoid changing sparsity structure
        K_dd_reg = K_dd.copy()
        K_dd_reg.setdiag(K_dd_reg.diagonal() + 1e-12)
        
        # Solve sparse system
        T_guyan_bottom = -spla.spsolve(K_dd_reg, K_da.toarray())
        if T_guyan_bottom.ndim == 1:
            T_guyan_bottom = T_guyan_bottom[:, np.newaxis]
        
        T_guyan = sp.vstack([I_a, sp.csr_matrix(T_guyan_bottom)])
        
        # Reorder rows
        reorder_indices = np.concatenate((keep_dofs, discard_dofs))
        T_guyan_reordered = sp.csr_matrix((T_guyan.shape[0], T_guyan.shape[1]))
        T_guyan_reordered[reorder_indices] = T_guyan
        
        return self.reduce(T_guyan_reordered)

    def reduce_dynamic(self, coordinates, frequency):
        """
        Perform Dynamic condensation on the sparse system

        Parameters
        ----------
        coordinates : CoordinateArray
            A list of coordinates to keep in the reduced system.
        frequency : float
            The frequency to preserve in the dynamic reduction.

        Returns
        -------
        System
            Reduced system as a dense System object.
        """
        if isinstance(coordinates, CoordinateArray):
            if not sp.issparse(self.transformation) or not np.allclose(self.transformation.toarray(), np.eye(*self.transformation.shape)):
                raise ValueError(
                    'Coordinates can only be specified with a CoordinateArray if the transformation is identity')
            keep_dofs = self.get_indices_by_coordinate(coordinates)
        else:
            keep_dofs = np.array(coordinates)
        discard_dofs = np.array([i for i in range(self.ndof) if i not in keep_dofs])
        I_a = sp.identity(keep_dofs.size, format='csr')
        D = self.stiffness - (2 * np.pi * frequency)**2 * self.mass
        D_dd = D[discard_dofs][:, discard_dofs]
        D_da = D[discard_dofs][:, keep_dofs]
        
        # Add small regularization to handle potential singularity
        # Use setdiag to avoid changing sparsity structure
        D_dd_reg = D_dd.copy()
        D_dd_reg.setdiag(D_dd_reg.diagonal() + 1e-12)
        
        # Solve sparse system
        T_dynamic_bottom = -spla.spsolve(D_dd_reg, D_da.toarray())
        if T_dynamic_bottom.ndim == 1:
            T_dynamic_bottom = T_dynamic_bottom[:, np.newaxis]
            
        T_dynamic = sp.vstack([I_a, sp.csr_matrix(T_dynamic_bottom)])
        
        # Reorder rows
        reorder_indices = np.concatenate((keep_dofs, discard_dofs))
        T_dynamic_reordered = sp.csr_matrix((T_dynamic.shape[0], T_dynamic.shape[1]))
        T_dynamic_reordered[reorder_indices] = T_dynamic
        
        return self.reduce(T_dynamic_reordered)

    def reduce_craig_bampton(self, connection_degrees_of_freedom: CoordinateArray,
                             num_fixed_base_modes: int,
                             return_shape_matrix: bool = False):
        """
        Computes a craig-bampton substructure model for the sparse system

        Parameters
        ----------
        connection_degrees_of_freedom : CoordinateArray
            Degrees of freedom to keep at the interface.
        num_fixed_base_modes : int
            Number of fixed-base modes to use in the reduction
        return_shape_matrix : bool, optional
            If true, return a set of shapes that represents the transformation
            in addition to the reduced system.  The default is False.

        Returns
        -------
        System
            Reduced system in craig-bampton form as a dense System object
        ShapeArray, optional
            Shapes representing the craig-bampton transformation if requested
        """
        # Construct craig bampton transformation
        if isinstance(connection_degrees_of_freedom, CoordinateArray):
            if not sp.issparse(self.transformation) or not np.allclose(self.transformation.toarray(), np.eye(*self.transformation.shape)):
                raise ValueError(
                    'Coordinates can only be specified with a CoordinateArray if the transformation is identity')
            connection_indices = self.get_indices_by_coordinate(connection_degrees_of_freedom)
        else:
            connection_indices = np.array(connection_degrees_of_freedom)
        other_indices = np.array([i for i in range(self.ndof) if i not in connection_indices])

        # Extract portions of the mass and stiffness matrices
        K_ii = self.K[other_indices][:, other_indices]
        M_ii = self.M[other_indices][:, other_indices]
        K_ib = self.K[other_indices][:, connection_indices]

        # Compute fixed interface modes using sparse eigenvalue solver
        max_modes = min(num_fixed_base_modes, K_ii.shape[0] - 2)
        sigma = 1.0
        lam, Phi_ii = spla.eigsh(K_ii, max_modes, M_ii, sigma=sigma, which='LM', tol=1e-3)
        
        # Sort by eigenvalue
        idx = np.argsort(lam)
        lam = lam[idx]
        Phi_ii = Phi_ii[:, idx]
        
        # Normalize the mode shapes
        lam[lam < 0] = 0
        M_ii_dense = M_ii.toarray()
        normalized_mass = np.diag(Phi_ii.T @ M_ii_dense @ Phi_ii)
        Phi_ii /= np.sqrt(normalized_mass)
        
        Z_bi = np.zeros((connection_indices.size, num_fixed_base_modes))
        
        # Compute constraint modes
        # Add small regularization to handle potential singularity
        # Use setdiag to avoid changing sparsity structure
        K_ii_reg = K_ii.copy()
        K_ii_reg.setdiag(K_ii_reg.diagonal() + 1e-12)
        Psi_ib = -spla.spsolve(K_ii_reg, K_ib.toarray())
        if Psi_ib.ndim == 1:
            Psi_ib = Psi_ib[:, np.newaxis]
        I_bb = np.eye(connection_indices.size)
        
        T_cb = np.block([[Phi_ii, Psi_ib],
                         [Z_bi, I_bb]])
        
        # Reorder rows
        reorder_indices = np.concatenate((other_indices, connection_indices))
        T_cb_reordered = np.zeros((T_cb.shape[0], T_cb.shape[1]))
        T_cb_reordered[reorder_indices] = T_cb
        
        if return_shape_matrix:
            from .sdynpy_shape import shape_array
            freq = np.sqrt(lam) / (2 * np.pi)
            all_freqs = np.concatenate((freq, np.zeros(connection_indices.size)))
            shapes = shape_array(self.coordinate, T_cb_reordered.T, all_freqs, comment1=['Fixed Base Mode {:}'.format(
                i + 1) for i in range(num_fixed_base_modes)] + ['Constraint Mode {:}'.format(str(dof)) for dof in connection_degrees_of_freedom])
            return self.reduce(T_cb_reordered), shapes
        else:
            return self.reduce(T_cb_reordered)

    def substructure_by_coordinate(self, dof_pairs, rcond=None,
                                   return_constrained_system=True):
        """
        Constrain the sparse system by connecting the specified degree of freedom pairs

        Parameters
        ----------
        dof_pairs : iterable of CoordinateArray
            Pairs of coordinates to be connected.  None can be passed instead of
            a second degree of freedom to constrain to ground
        rcond : float, optional
            Condition threshold to use for the nullspace calculation on the
            constraint matrix. The default is None.
        return_constrained_system : bool, optional
            If true, apply the constraint matrix and return the constrained
            system, otherwise simply return the constraint matrix. The default
            is True.

        Returns
        -------
        np.ndarray or System
            Returns a System object with the constraints applied if
            `return_constrained_system` is True, otherwise just return the
            constraint matrix.
        """
        constraint_matrix = []
        for constraint_dof_0, constraint_dof_1 in dof_pairs:
            constraint = self.transformation_matrix_at_coordinates(constraint_dof_0)
            if constraint_dof_1 is not None:
                constraint -= self.transformation_matrix_at_coordinates(constraint_dof_1)
            constraint_matrix.append(constraint.toarray())  # Convert to dense for concatenation
        constraint_matrix = np.concatenate(constraint_matrix, axis=0)
        if return_constrained_system:
            return self.constrain(constraint_matrix, rcond)
        else:
            return constraint_matrix

    def substructure_by_shape(self, constraint_shapes, connection_dofs_0,
                              connection_dofs_1=None, rcond=None,
                              return_constrained_system=True):
        """
        Constrain the sparse system using a set of shapes in a least-squares sense.

        Parameters
        ----------
        constraint_shapes : ShapeArray
            An array of shapes to use as the basis for the constraints
        connection_dofs_0 : CoordinateArray
            Array of coordinates to use in the constraints
        connection_dofs_1 : CoordinateArray, optional
            Array of coordinates to constrain to the coordinates in
            `connection_dofs_0`. If not specified, the `connection_dofs_0`
            degrees of freedom will be constrained to ground.
        rcond : float, optional
            Condition threshold on the nullspace calculation. The default is None.
        return_constrained_system : bool, optional
            If true, apply the constraint matrix and return the constrained
            system, otherwise simply return the constraint matrix. The default
            is True.

        Returns
        -------
        np.ndarray or System
            Returns a System object with the constraints applied if
            `return_constrained_system` is True, otherwise just return the
            constraint matrix.
        """
        shape_matrix_0 = constraint_shapes[connection_dofs_0].T
        transform_matrix_0 = self.transformation_matrix_at_coordinates(connection_dofs_0).toarray()
        constraint_matrix = np.linalg.lstsq(shape_matrix_0, transform_matrix_0)[0]
        if connection_dofs_1 is not None:
            shape_matrix_1 = constraint_shapes[connection_dofs_1].T
            transform_matrix_1 = self.transformation_matrix_at_coordinates(connection_dofs_1).toarray()
            constraint_matrix -= np.linalg.lstsq(shape_matrix_1, transform_matrix_1)[0]
        if return_constrained_system:
            return self.constrain(constraint_matrix, rcond)
        else:
            return constraint_matrix

    def save(self, filename):
        """
        Saves the sparse system to a file

        Parameters
        ----------
        filename : str
            Name of the file in which the system will be saved.
        """
        # Save sparse matrices in their sparse format to avoid memory issues
        import pickle
        
        save_data = {
            'mass': self.mass,
            'stiffness': self.stiffness,
            'damping': self.damping,
            'transformation': self.transformation,
            'coordinate': self.coordinate,
            'sparse_format': True
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, filename):
        """
        Load a sparse system from a file

        Parameters
        ----------
        filename : str
            Name of the file from which the system will be loaded.

        Returns
        -------
        SystemSparse
            A sparse system consisting of the mass, stiffness, damping, and transformation
            in the file
        """
        import pickle
        
        try:
            # Try to load as pickle first (new format)
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict) and data.get('sparse_format', False):
                return cls(data['coordinate'], 
                          data['mass'],
                          data['stiffness'], 
                          data['damping'], 
                          data['transformation'])
            else:
                raise ValueError("File does not contain sparse system data")
                
        except (pickle.UnpicklingError, FileNotFoundError):
            # Fallback to old numpy format for backward compatibility
            try:
                data = np.load(filename)
                return cls(data['coordinate'].view(CoordinateArray), 
                          sp.csr_matrix(data['mass']),
                          sp.csr_matrix(data['stiffness']), 
                          sp.csr_matrix(data['damping']), 
                          sp.csr_matrix(data['transformation']))
            except Exception as e:
                raise ValueError(f"Could not load sparse system from {filename}: {e}")



    @classmethod
    def concatenate(cls, systems, coordinate_node_offset=0):
        """
        Combine multiple sparse systems together

        Parameters
        ----------
        systems : iterable of SystemSparse objects
            Iterable of SystemSparse objects that will be concatenated.  Matrices will be
            assembled in block diagonal format
        coordinate_node_offset : int, optional
            Offset applied to the coordinates so the nodes do not overlap.
            The default is 0.

        Returns
        -------
        SystemSparse
            A sparse system consisting of the combination of the provided systems.
        """
        coordinates = [system.coordinate.copy() for system in systems]
        if coordinate_node_offset != 0:
            for i in range(len(coordinates)):
                coordinates[i].node += coordinate_node_offset * (i + 1)
        all_coordinates = np.concatenate(coordinates)
        
        # Use scipy's block_diag for sparse matrices
        return cls(all_coordinates,
                   sp.block_diag([system.mass for system in systems], format='csr'),
                   sp.block_diag([system.stiffness for system in systems], format='csr'),
                   sp.block_diag([system.damping for system in systems], format='csr'),
                   sp.block_diag([system.transformation for system in systems], format='csr'))

    # Methods that are not implemented for sparse systems
    def beam(self, *args, **kwargs):
        """Not implemented for sparse systems"""
        raise NotImplementedError("beam() method not implemented for SystemSparse. Use System.beam() instead.")

    def to_state_space(self, *args, **kwargs):
        """Not implemented for sparse systems"""
        raise NotImplementedError("to_state_space() method not implemented for SystemSparse")

    def time_integrate(self, *args, **kwargs):
        """Not implemented for sparse systems"""
        raise NotImplementedError("time_integrate() method not implemented for SystemSparse")

    def frequency_response(self, *args, **kwargs):
        """Not implemented for sparse systems"""
        raise NotImplementedError("frequency_response() method not implemented for SystemSparse")

    @classmethod
    def from_exodus_superelement(cls, *args, **kwargs):
        """Not implemented for sparse systems"""
        raise NotImplementedError("from_exodus_superelement() method not implemented for SystemSparse")

    def simulate_test(self, *args, **kwargs):
        """Not implemented for sparse systems"""
        raise NotImplementedError("simulate_test() method not implemented for SystemSparse")
