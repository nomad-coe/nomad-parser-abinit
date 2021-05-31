#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np            # pylint: disable=unused-import
import typing                 # pylint: disable=unused-import
from nomad.metainfo import (  # pylint: disable=unused-import
    MSection, MCategory, Category, Package, Quantity, Section, SubSection, SectionProxy,
    Reference
)
from nomad.metainfo.legacy import LegacyDefinition

from abinitparser.metainfo import abinit_autogenerated
from nomad.datamodel.metainfo import public

m_package = Package(
    name='abinit_nomadmetainfo_json',
    description='None',
    a_legacy=LegacyDefinition(name='abinit.nomadmetainfo.json'))


class x_abinit_section_stress_tensor(MSection):
    '''
    Section describing the stress tensor
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_abinit_section_stress_tensor'))

    x_abinit_stress_tensor_xx = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        xx component of the stress tensor
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_stress_tensor_xx'))

    x_abinit_stress_tensor_yy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        yy component of the stress tensor
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_stress_tensor_yy'))

    x_abinit_stress_tensor_zz = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        zz component of the stress tensor
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_stress_tensor_zz'))

    x_abinit_stress_tensor_zy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        zy component of the stress tensor
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_stress_tensor_zy'))

    x_abinit_stress_tensor_zx = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        zx component of the stress tensor
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_stress_tensor_zx'))

    x_abinit_stress_tensor_yx = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        yx component of the stress tensor
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_stress_tensor_yx'))


class x_abinit_section_dataset_header(MSection):
    '''
    -
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_abinit_section_dataset_header'))

    x_abinit_dataset_number = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Dataset number
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_dataset_number'))

    x_abinit_vprim_1 = Quantity(
        type=str,
        shape=[],
        description='''
        Primitive axis 1
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_vprim_1'))

    x_abinit_vprim_2 = Quantity(
        type=str,
        shape=[],
        description='''
        Primitive axis 2
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_vprim_2'))

    x_abinit_vprim_3 = Quantity(
        type=str,
        shape=[],
        description='''
        Primitive axis 3
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_vprim_3'))


class x_abinit_section_var(MSection):
    '''
    -
    '''

    m_def = Section(validate=False, a_legacy=LegacyDefinition(name='x_abinit_section_var'))

    x_abinit_vardtset = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Variable dataset number
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_vardtset'))

    x_abinit_varname = Quantity(
        type=str,
        shape=[],
        description='''
        Variable name
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_varname'))

    x_abinit_varvalue = Quantity(
        type=str,
        shape=[],
        description='''
        Variable value
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_varvalue'))

    x_abinit_vartruncation = Quantity(
        type=str,
        shape=[],
        description='''
        Variable truncation length
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_vartruncation'))


class section_run(public.section_run):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_run'))

    x_abinit_parallel_compilation = Quantity(
        type=str,
        shape=[],
        description='''
        Parallel or sequential compilation
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_parallel_compilation'))

    x_abinit_start_date = Quantity(
        type=str,
        shape=[],
        description='''
        Start date as string
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_start_date'))

    x_abinit_start_time = Quantity(
        type=str,
        shape=[],
        description='''
        Start time as string
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_start_time'))

    x_abinit_input_file = Quantity(
        type=str,
        shape=[],
        description='''
        Input file name
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_input_file'))

    x_abinit_output_file = Quantity(
        type=str,
        shape=[],
        description='''
        Output file name
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_output_file'))

    x_abinit_input_files_root = Quantity(
        type=str,
        shape=[],
        description='''
        Root for input files
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_input_files_root'))

    x_abinit_output_files_root = Quantity(
        type=str,
        shape=[],
        description='''
        Root for output files
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_output_files_root'))

    x_abinit_total_cpu_time = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Total CPU time
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_total_cpu_time'))

    x_abinit_total_wallclock_time = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Total wallclock time
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_total_wallclock_time'))

    x_abinit_completed = Quantity(
        type=str,
        shape=[],
        description='''
        Message that the calculation was completed
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_completed'))

    x_abinit_section_var = SubSection(
        sub_section=SectionProxy('x_abinit_section_var'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_abinit_section_var'))


class section_method(public.section_method):
    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_method'))

    x_abinit_tolvrs = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        `TOLerance on the potential V(r) ReSidual`:
        Sets a tolerance for potential residual that, when reached, will cause
        one SCF cycle to stop (and ions to be moved). If set to zero, this
        stopping condition is ignored. Instead, refer to other tolerances, such
        as toldfe, tolwfr.
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_tolvrs'))

    x_abinit_tolwfr = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        TOLerance on WaveFunction squared Residual:
        Specifies the threshold on WaveFunction squared Residuals;
        it gives a convergence tolerance for the largest squared residual
        for any given band.
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_tolwfr'))

    x_abinit_istwfk = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
         Integer for choice of STorage of WaveFunction at each k point;
        Controls the way the wavefunction for each k-point is stored inside ABINIT,
        in reciprocal space, according to time-reversal symmetry properties.
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_istwfk'))

    x_abinit_iscf = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ABINIT variable Integer for Self-Consistent-Field cycles
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_iscf'))


class section_system(public.section_system):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_system'))

    x_abinit_atom_xcart_final = Quantity(
        type=str,
        shape=[],
        description='''
        Cartesian coordinates of an atom at the end of the dataset
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_atom_xcart_final'))

    x_abinit_atom_xcart = Quantity(
        type=str,
        shape=[],
        description='''
        Cartesian coordinates of an atom at the end of a single configuration calculation
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_atom_xcart'))


class section_single_configuration_calculation(public.section_single_configuration_calculation):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='section_single_configuration_calculation'))

    x_abinit_magnetisation = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Total magnetisation.
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_magnetisation'))

    x_abinit_fermi_energy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Fermi energy.
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_fermi_energy'))

    x_abinit_single_configuration_calculation_converged = Quantity(
        type=str,
        shape=[],
        description='''
        Determines whether a single configuration calculation is converged.
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_single_configuration_calculation_converged'))

    x_abinit_atom_force = Quantity(
        type=str,
        shape=[],
        description='''
        Force acting on an atom at the end of a single configuration calculation
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_atom_force'))

    x_abinit_atom_force_final = Quantity(
        type=np.dtype(np.float64),
        unit='newton',
        shape=[],
        description='''
        Force acting on an atom at the end of the dataset
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_atom_force_final'))

    x_abinit_energy_ewald = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Ewald energy
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_energy_ewald'))

    x_abinit_energy_psp_core = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Pseudopotential core energy
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_energy_psp_core'))

    x_abinit_energy_psp_local = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Local pseudopotential energy
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_energy_psp_local'))

    x_abinit_energy_psp_nonlocal = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Non-local pseudopotential energy
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_energy_psp_nonlocal'))

    x_abinit_energy_internal = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Internal energy
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_energy_internal'))

    x_abinit_energy_ktentropy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -kT*entropy
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_energy_ktentropy'))

    x_abinit_energy_band = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        Band energy
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_energy_band'))

    x_abinit_section_stress_tensor = SubSection(
        sub_section=SectionProxy('x_abinit_section_stress_tensor'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_abinit_section_stress_tensor'))

    x_abinit_unit_cell_volume = Quantity(
        type=np.dtype(np.float64),
        unit='meter**3',
        shape=[],
        description='''
        Unit cell volume
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_unit_cell_volume'))


class x_abinit_section_dataset(abinit_autogenerated.x_abinit_section_dataset):

    m_def = Section(validate=False, extends_base_section=True, a_legacy=LegacyDefinition(name='x_abinit_section_dataset'))

    x_abinit_geometry_optimization_converged = Quantity(
        type=str,
        shape=[],
        description='''
        Determines whether a geometry optimization is converged.
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_geometry_optimization_converged'))

    x_abinit_eig_filename = Quantity(
        type=str,
        shape=[],
        description='''
        Name of file where the eigenvalues were written to.
        ''',
        a_legacy=LegacyDefinition(name='x_abinit_eig_filename'))

    x_abinit_section_dataset_header = SubSection(
        sub_section=SectionProxy('x_abinit_section_dataset_header'),
        repeats=True,
        a_legacy=LegacyDefinition(name='x_abinit_section_dataset_header'))


m_package.__init_metainfo__()
