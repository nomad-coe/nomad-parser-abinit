#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
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

import pytest
import numpy as np

from nomad.datamodel import EntryArchive
from abinitparser import AbinitParser


@pytest.fixture(scope='module')
def parser():
    return AbinitParser()


def test_scf(parser):
    archive = EntryArchive()
    parser.parse('tests/data/Si/Si.out', archive, None)

    sec_run = archive.section_run[0]
    assert sec_run.program_version == '7.8.2'
    assert sec_run.x_abinit_total_cpu_time == 1.4
    assert sec_run.run_clean_end
    assert pytest.approx(sec_run.time_run_date_start.magnitude, 1.46712528e+09)
    sec_dataset = sec_run.x_abinit_section_dataset
    assert len(sec_dataset) == 1
    assert len(sec_dataset[0].x_abinit_section_input[0].x_abinit_var_symrel) == 432
    assert sec_dataset[0].x_abinit_section_input[0].x_abinit_var_znucl[0] == 14.
    assert pytest.approx(sec_run.section_basis_set_cell_dependent[0].basis_set_planewave_cutoff.magnitude, 3.48779578e-17)

    sec_method = sec_run.section_method[0]
    assert sec_method.scf_max_iteration == 10.
    assert pytest.approx(sec_method.scf_threshold_energy_change.magnitude, 4.35974472e-24)
    assert sec_method.section_method_basis_set[0].method_basis_set_kind == 'wavefunction'
    assert sec_method.section_XC_functionals[0].XC_functional_name == 'LDA_XC_TETER93'

    sec_system = sec_run.section_system[0]
    assert sec_system.atom_labels == ['Si', 'Si']
    assert False not in sec_system.configuration_periodic_dimensions
    assert pytest.approx(sec_system.atom_positions[1][1].magnitude, 1.346756e-10)
    assert pytest.approx(sec_system.lattice_vectors[2][0].magnitude, 2.693512e-10)

    sec_scc = sec_run.section_single_configuration_calculation[0]
    assert pytest.approx(sec_scc.energy_total.magnitude, -3.86544728e-17)
    assert np.max(sec_scc.atom_forces_raw.magnitude) == 0.
    assert pytest.approx(sec_scc.stress_tensor[2][2].magnitude, -5.60539974e+08)
    assert pytest.approx(sec_scc.energy_reference_fermi[0].magnitude, 8.4504932e-19)
    assert pytest.approx(sec_scc.x_abinit_energy_kinetic, 1.3343978e-17)
    assert len(sec_scc.section_scf_iteration) == 5
    sec_eig = sec_scc.section_eigenvalues[0]
    assert pytest.approx(sec_scc.section_scf_iteration[1].energy_total_scf_iteration.magnitude, -3.86541222e-17)
    assert np.shape(sec_eig.eigenvalues_values) == (1, 2, 5)
    assert pytest.approx(sec_eig.eigenvalues_values[0][1][2].magnitude, 8.4504932e-19)
    assert sec_eig.eigenvalues_kpoints[0][0] == -0.25


def test_relax(parser):
    archive = EntryArchive()
    parser.parse('tests/data/H2/H2.out', archive, None)

    assert len(archive.section_run[0].x_abinit_section_dataset) == 2
    sec_sccs = archive.section_run[0].section_single_configuration_calculation
    assert len(sec_sccs) == 5
    assert len(sec_sccs[2].section_scf_iteration) == 5
    assert pytest.approx(sec_sccs[3].energy_total.magnitude, -4.93984603e-18)
    assert pytest.approx(sec_sccs[4].section_scf_iteration[1].energy_total_scf_iteration.magnitude, -2.13640055e-18)


def test_dos(parser):
    archive = EntryArchive()
    parser.parse('tests/data/Fe/Fe.out', archive, None)

    sec_sccs = archive.section_run[0].section_single_configuration_calculation
    assert len(sec_sccs) == 2
    assert np.shape(sec_sccs[0].section_eigenvalues[0].eigenvalues_values) == (1, 6, 8)
    assert np.shape(sec_sccs[1].section_eigenvalues[0].eigenvalues_values) == (2, 6, 8)
    assert np.shape(sec_sccs[0].section_dos[0].dos_values) == (1, 1601)
    assert np.shape(sec_sccs[1].section_dos[0].dos_values) == (2, 1601)
    assert pytest.approx(sec_sccs[0].section_dos[0].dos_energies[70].magnitude, -3.18261365e-18)
    assert pytest.approx(sec_sccs[0].section_dos[0].dos_values[0][151], 3.5095398518405453e-15)
    assert pytest.approx(sec_sccs[1].section_dos[0].dos_values[1][180], 1.50393389e-14)
    assert pytest.approx(sec_sccs[1].section_dos[0].dos_values[1][180], 1.50393389e-14)
    assert pytest.approx(sec_sccs[0].section_dos[0].dos_integrated_values[0][457], 3.44346629e-29)
    assert pytest.approx(sec_sccs[1].section_dos[0].dos_integrated_values[0][1025], 1.65377776e-28)
