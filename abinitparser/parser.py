# Copyright 2016-2018 Micael Oliveira, Fawzi Mohamed
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

from builtins import object
from nomadcore.caching_backend import CachingLevel
from nomadcore.simple_parser import SimpleMatcher as SM
from nomadcore.simple_parser import mainFunction
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
from nomadcore import parser_backend
from nomadcore.unit_conversion import unit_conversion
from ase.data import chemical_symbols
from abinitparser.AbinitXC import ABINIT_NATIVE_IXC, ABINIT_LIBXC_IXC
import numpy as np
import re
import os
import logging
import time
import sys

try:
    basestring
except NameError:
    basestring = str

logger = logging.getLogger("nomad.ABINITParser")

parserInfo = {
  "name": "ABINIT_parser",
  "version": "1.0"
}

ABINIT_GEO_OPTIMIZATION = {
    1: "viscous_damped_md",
    2: "bfgs",
    3: "bfgs",
    4: "conjugate_gradient",
    5: "steepest_descent",
    7: "quenched_md",
    10: "dic_bfgs",
    11: "dic_bfgs",
    20: "diis"
}

class ABINITContext(object):
    """context for the sample parser"""

    def __init__(self):
        self.parser = None
        self.current_dataset = None
        self.abinitVars = None
        self.input = None
        self.initial_simulation_cell = None
        self.inputGIndex = None
        self.methodGIndex = None
        self.systemGIndex = None
        self.samplingGIndex = None
        self.basisGIndex = None
        self.frameSequence = []

    def initialize_values(self):
        """allows to reset values if the same superContext is used to parse different files"""
        self.current_dataset = 0
        # Initialize dict to store Abinit variables. Two of them are created by default:
        #  - dataset "0", which will contain the values that are common to all datasets
        #  - dataset "1", as this is the default dataset number used by Abinit when the user
        #    does not specify the dataset number
        self.abinitVars = {key: {} for key in [0, 1]}
        self.input = None
        self.initial_simulation_cell = None
        self.inputGIndex = None
        self.methodGIndex = None
        self.systemGIndex = None
        self.samplingGIndex = None
        self.basisGIndex = None
        self.frameSequence = []

    def startedParsing(self, filename, parser):
        """called when parsing starts"""
        self.parser = parser
        # allows to reset values if the same superContext is used to parse different files
        self.initialize_values()

    def onClose_section_run(self, backend, gIndex, section):
        """Trigger called when section_run is closed.
        """
        if section["x_abinit_completed"] is not None:
            backend.addValue("run_clean_end", True)
        # Convert date and time to epoch time
        if (section["x_abinit_start_date"] is not None) and (section["x_abinit_start_time"] is not None):
            abi_time = time.strptime(str("%s %s") % (section["x_abinit_start_date"][-1],
                                                     section["x_abinit_start_time"][-1]), "%a %d %b %Y %Hh%M")
            backend.addValue("time_run_date_start", time.mktime(abi_time))

    def onOpen_x_abinit_section_dataset(self, backend, gIndex, section):
        """Trigger called when x_abinit_section_dataset is opened.
        """
        self.samplingGIndex = backend.openSection("section_sampling_method")

    def onClose_x_abinit_section_dataset(self, backend, gIndex, section):
        """Trigger called when x_abinit_section_dataset is closed.
        """
        if len(self.frameSequence) > 1:
            frameGIndex = backend.openSection("section_frame_sequence")
            if section["x_abinit_geometry_optimization_converged"] is not None:
                if section["x_abinit_geometry_optimization_converged"][-1] == "are converged":
                    backend.addValue("geometry_optimization_converged", True)
                else:
                    backend.addValue("geometry_optimization_converged", False)
            backend.closeSection("section_frame_sequence", frameGIndex)
        self.frameSequence = []
        backend.closeSection("section_sampling_method", self.samplingGIndex)

    def onClose_section_frame_sequence(self, backend, gIndex, section):
        """Trigger called when section_framce_sequence is closed.
        """
        backend.addValue("number_of_frames_in_sequence", len(self.frameSequence))
        backend.addArrayValues("frame_sequence_local_frames_ref", np.array(self.frameSequence))
        backend.addValue("frame_sequence_to_sampling_ref", self.samplingGIndex)

    def onClose_section_sampling_method(self, backend, gIndex, section):
        """Trigger called when section_sampling_method is closed.
        """
        if self.input["x_abinit_var_ionmov"] is not None:
            ionmov = self.input["x_abinit_var_ionmov"][-1]
            if ionmov in [2, 3, 4, 5, 7, 10, 11, 20] or (ionmov == 1 and self.input["x_abinit_var_vis"] > 0.0):
                sampling_method = "geometry_optimization"
                backend.addValue("geometry_optimization_method", ABINIT_GEO_OPTIMIZATION[ionmov])
            elif ionmov in [6, 8, 12, 13, 14, 23] or (ionmov == 1 and self.input["x_abinit_var_vis"] == 0.0):
                sampling_method = "molecular_dynamics"
            elif ionmov == 9:
                sampling_method = "langevin_dynamics"
            else:
                sampling_method = ""
            backend.addValue("sampling_method", sampling_method)

        if self.input["x_abinit_var_tolmxf"] is not None:
            backend.addValue("geometry_optimization_threshold_force",
                             unit_conversion.convert_unit(self.input["x_abinit_var_tolmxf"][-1], "hartree / bohr"))
        if self.input["x_abinit_var_tolmxde"] is not None:
            backend.addValue("geometry_optimization_energy_change",
                             unit_conversion.convert_unit(self.input["x_abinit_var_tolmxde"][-1], "hartree"))

    def onOpen_section_single_configuration_calculation(self, backend, gIndex, section):
        """Trigger called when section_single_configuration_calculation is opened.
        """
        self.systemGIndex = backend.openSection("section_system")
        backend.addValue("single_configuration_calculation_to_system_ref", self.systemGIndex)
        backend.addValue("single_configuration_to_calculation_method_ref", self.methodGIndex)
        self.frameSequence.append(gIndex)

    def onClose_section_single_configuration_calculation(self, backend, gIndex, section):
        """Trigger called when section_single_configuration_calculation is closed.
        """
        backend.closeSection("section_system", self.systemGIndex)

        converged = False
        if section["x_abinit_single_configuration_calculation_converged"] is not None:
            if section["x_abinit_single_configuration_calculation_converged"][-1] == "is converged" or \
               section["x_abinit_single_configuration_calculation_converged"][-1] == "converged" or \
               section["x_abinit_single_configuration_calculation_converged"][-1] == "are converged":
                converged = True
        backend.addValue("single_configuration_calculation_converged", converged)

        if section["x_abinit_energy_xc"] is not None:
            backend.addValue("energy_XC", unit_conversion.convert_unit(section["x_abinit_energy_xc"][-1], "hartree"))
        if section["x_abinit_energy_kinetic"] is not None:
            backend.addValue("electronic_kinetic_energy",
                             unit_conversion.convert_unit(section["x_abinit_energy_kinetic"][-1], "hartree"))

        if section["energy_total"] is None and section["x_abinit_energy_total"] is not None:
            backend.addValue("energy_total",
                             unit_conversion.convert_unit(section["x_abinit_energy_total"][-1], "hartree"))

        if section["x_abinit_atom_force"] is not None:
            atom_forces_list = section["x_abinit_atom_force"]
        elif section["x_abinit_atom_force_final"] is not None:
            atom_forces_list = section["x_abinit_atom_force_final"]
        else:
            atom_forces_list = None

        if atom_forces_list is not None:
            atom_forces = backend.arrayForMetaInfo("atom_forces_raw", [self.input["x_abinit_var_natom"][-1],3])
            n_atom = 0
            for force_string in atom_forces_list:
                for dir in range(3):
                    atom_forces[n_atom, dir] = unit_conversion.convert_unit(float(force_string.split()[dir]), "hartree / bohr")
                n_atom += 1
            backend.addArrayValues("atom_forces_raw", atom_forces)

        if section["x_abinit_fermi_energy"] is not None:
            backend.addArrayValues("energy_reference_fermi",
                                   np.array([unit_conversion.convert_unit(section["x_abinit_fermi_energy"][-1], "hartree")]))

    def onClose_section_eigenvalues(self, backend, gIndex, section):
        """Trigger called when section_eigenvalues is closed.
        """
        nkpt = int(self.input["x_abinit_var_nkpt"][-1])
        nspin = int(self.input["x_abinit_var_nsppol"][-1])
        nband = int(self.input["x_abinit_var_nband"][-1][0][0])
        for ispin in range(nspin):
            for ikpt in range(nkpt):
                if self.input["x_abinit_var_nband"][-1][ikpt][ispin] != nband:
                    nband = 0
                    break

        if nband == 0:
            logger.warn("Number of bands in this calculation is k-point dependent.")
        else:
            if section["x_abinit_eigenvalues"] is not None:
                if len(section["x_abinit_eigenvalues"]) == 1:
                    ev_string = section["x_abinit_eigenvalues"][0]
                else:
                    ev_string = " ".join(section["x_abinit_eigenvalues"])
                abi_eigenvalues = ev_string.split()
            else:
                abi_eigenvalues = []
                logger.warn("Eigenvalues are not available.")
            eigenvalues = np.array([unit_conversion.convert_unit(float(x), "hartree") for x in abi_eigenvalues])

            if section["x_abinit_occupations"] is not None:
                if len(section["x_abinit_occupations"]) == 1:
                    occs_string = section["x_abinit_occupations"][0]
                else:
                    occs_string = " ".join(section["x_abinit_occupations"])
                abi_occs = occs_string.split()
            elif self.input["x_abinit_var_occ"][-1] is not None:
                abi_occs = self.input["x_abinit_var_occ"][-1]
            else:
                abi_occs = []
                logger.warn("Occupations are not available.")
            occupations = np.array([float(x) for x in abi_occs])

            abi_kpoints = []
            if section["x_abinit_kpt"] is not None:
                for kpt in section["x_abinit_kpt"]:
                    abi_kpoints.append([float(x) for x in kpt.split()])
            elif self.input["x_abinit_var_kpt"] is not None:
                for ikpt in range(nkpt):
                    abi_kpoints.append([float(x) for x in self.input["x_abinit_var_kpt"][-1][ikpt]])
            else:
                logger.warn("K-points are not available.")
            kpoints = np.array(abi_kpoints)

            if len(kpoints) == nkpt and len(eigenvalues) == nband*nspin*nkpt and len(occupations) == nband * nspin * nkpt:
                backend.addValue("eigenvalues_kind", "normal")
                backend.addValue("number_of_eigenvalues", nband)
                backend.addValue("number_of_eigenvalues_kpoints", nkpt)
                backend.addArrayValues("eigenvalues_kpoints", kpoints)
                backend.addArrayValues("eigenvalues_values", eigenvalues.reshape([nspin, nkpt, nband]))
                backend.addArrayValues("eigenvalues_occupation", occupations.reshape([nspin, nkpt, nband]))

    def onOpen_x_abinit_section_dataset_header(self, backend, gIndex, section):
        """Trigger called when x_abinit_section_dataset is opened.
        """
        self.inputGIndex = backend.openSection("x_abinit_section_input")

    def onClose_x_abinit_section_dataset_header(self, backend, gIndex, section):
        """Trigger called when x_abinit_section_dataset is closed.
        """
        self.current_dataset = section["x_abinit_dataset_number"][-1]
        backend.closeSection("x_abinit_section_input", self.inputGIndex)

        self.basisGIndex = backend.openSection("section_basis_set_cell_dependent")
        backend.addValue("basis_set_cell_dependent_kind", "plane_waves")
        if self.input["x_abinit_var_ecut"] is not None:
            backend.addValue("basis_set_planewave_cutoff",
                             unit_conversion.convert_unit(self.input["x_abinit_var_ecut"][-1], 'hartree'))
        backend.addValue("basis_set_cell_dependent_name",
                         "PW_%s"%(unit_conversion.convert_unit(self.input["x_abinit_var_ecut"][-1], 'hartree', 'rydberg')))
        backend.closeSection("section_basis_set_cell_dependent", self.basisGIndex)

        self.methodGIndex = backend.openSection("section_method")
        backend.closeSection("section_method", self.methodGIndex)

        self.initial_simulation_cell = []
        for axis in [1, 2, 3]:
            # FAIRD edit (dts@physik.hu-berlin.de), the split is failing for negative
            # decimals in the third column. It gives "2.7 2.7-2.7" for some reason.
            # We need to add a whitespace to split nicely.
            self.initial_simulation_cell.append(
               section["x_abinit_vprim_%s" % axis][-1].replace('-',' -').split())

    def onClose_section_method(self, backend, gIndex, section):
        """Trigger called when section_method is closed.
        """
        backend.addValue("stress_tensor_method", "analytic")
        backend.addValue("number_of_spin_channels", self.input["x_abinit_var_nsppol"][-1])
        backend.addValue("scf_max_iteration", self.input["x_abinit_var_nstep"][-1])
        if self.input["x_abinit_var_toldfe"] is not None:
            backend.addValue("scf_threshold_energy_change",
                             unit_conversion.convert_unit(self.input["x_abinit_var_toldfe"][-1], 'hartree'))
        backend.addValue("self_interaction_correction_method", "")
        if self.input["x_abinit_var_occopt"][-1] == 3:
            smear_kind = "fermi"
        elif self.input["x_abinit_var_occopt"][-1] == 4 or self.input["x_abinit_var_occopt"] == 5:
            smear_kind = "marzari-vanderbilt"
        elif self.input["x_abinit_var_occopt"][-1] == 6:
            smear_kind = "methfessel-paxton"
        elif self.input["x_abinit_var_occopt"][-1] == 7:
            smear_kind = "gaussian"
        elif self.input["x_abinit_var_occopt"][-1] == 8:
            logger.error("Illegal value for Abinit input variable occopt")
            smear_kind = ""
        else:
            smear_kind = ""
        backend.addValue("smearing_kind", smear_kind)
        if self.input["x_abinit_var_tsmear"] is not None:
            backend.addValue("smearing_width",
                             unit_conversion.convert_unit(self.input["x_abinit_var_tsmear"][-1], 'hartree'))

        gIndex = backend.openSection("section_method_basis_set")
        backend.addValue("method_basis_set_kind", "wavefunction")
        backend.addValue("mapping_section_method_basis_set_cell_associated", self.basisGIndex)
        backend.closeSection("section_method_basis_set", gIndex)

        backend.addValue("electronic_structure_method", "DFT")
        ixc = int(self.input["x_abinit_var_ixc"][-1])
        if ixc >= 0:
            xc_functionals = ABINIT_NATIVE_IXC[str(ixc)]
        else:
            xc_functionals = []
            functional1 = -ixc//1000
            if functional1 > 0:
                xc_functionals.append(ABINIT_LIBXC_IXC[str(functional1)])
            functional2 = -ixc - (-ixc//1000)*1000
            if functional2 > 0:
                xc_functionals.append(ABINIT_LIBXC_IXC[str(functional2)])

        if xc_functionals is not None:
            for xc_functional in xc_functionals:
                gIndex = backend.openSection('section_XC_functionals')
                for key, value in sorted(xc_functional.items()):
                    if isinstance(value, (list, dict)):
                        backend.addValue(key, value)
                    else:
                        backend.addValue(key, value)
                backend.closeSection('section_XC_functionals', gIndex)

    def onClose_section_system(self, backend, gIndex, section):
        """Trigger called when section_system is closed.
        """
        species_count = {}
        for z in self.input["x_abinit_var_znucl"][-1]:
            species_count[chemical_symbols[int(z)]] = 0

        atom_types = []
        for z in self.input["x_abinit_var_znucl"][-1]:
            symbol = chemical_symbols[int(z)]
            species_count[symbol] += 1
            if species_count[symbol] > 1:
               atom_type = symbol + str(species_count[symbol])
            else:
               atom_type = symbol
            atom_types.append(atom_type)

        # Grabs an array with the dtype of meta name atom_labels and
        # shape of self.input('x_abinit..').
        atom_labels = backend.arrayForMetaInfo("atom_labels", self.input["x_abinit_var_natom"][-1])

        for atom_index in range(self.input["x_abinit_var_natom"][-1]):
            atom_labels[atom_index] = atom_types[self.input["x_abinit_var_typat"][-1][atom_index] - 1]

        backend.addArrayValues("atom_labels", atom_labels)

        if section["x_abinit_atom_xcart"] is not None:
            atom_xcart_list = section["x_abinit_atom_xcart"]
            xcart_unit = "bohr"
        elif section["x_abinit_atom_xcart_final"] is not None:
            atom_xcart_list = section["x_abinit_atom_xcart_final"]
            xcart_unit = "angstrom"
        else:
            atom_xcart_list = None
        n_atom = self.input["x_abinit_var_natom"][-1]
        if atom_xcart_list is not None:
            atom_xcart = backend.arrayForMetaInfo("atom_positions", [n_atom, 3])
            for iatom in range(n_atom):
                for dir in range(3):
                    atom_xcart[iatom, dir] = float(atom_xcart_list[iatom].split()[dir])
        else:
            xcart_unit = "bohr"
            if self.input["x_abinit_var_xcart"] is not None:
                atom_xcart = self.input["x_abinit_var_xcart"][-1]
            elif n_atom == 1:
                atom_xcart = np.array([[0, 0, 0]])
            else:
                logger.error("Positions of atoms is not available")
        for iatom in range(n_atom):
            for dir in range(3):
                atom_xcart[iatom, dir] = unit_conversion.convert_unit(atom_xcart[iatom, dir], xcart_unit)
        backend.addArrayValues("atom_positions", atom_xcart)


        backend.addArrayValues("configuration_periodic_dimensions", np.array([True, True, True]))

        backend.addValue("number_of_atoms", self.input["x_abinit_var_natom"][-1])

        #backend.addValue("space_group_3D_number", self.input["x_abinit_var_spgroup"][-1]) # leave this to the normalizer...

        vprim = backend.arrayForMetaInfo("simulation_cell", [3, 3])
        for axis in [1, 2, 3]:
            if section["x_abinit_vprim_%s"%axis] is None:
                vprim[axis-1] = self.initial_simulation_cell[axis-1]
            else:
                vprim[axis-1] = section["x_abinit_vprim_%s"%axis][-1].split()
            for component in range(3):
                vprim[axis-1][component] = unit_conversion.convert_unit(vprim[axis-1][component], 'bohr')
        backend.addArrayValues("simulation_cell", vprim)

    def onOpen_x_abinit_section_input(self, backend, gIndex, section):
        """Trigger called when x_abinit_section_input is opened.
        """
        self.input = section

    def onClose_x_abinit_section_input(self, backend, gIndex, section):
        """Trigger called when x_abinit_section_input is closed.
        """
        dataset_vars = {}
        for varname in backend.superBackend.metaInfoEnv().infoKinds.keys():
            if "x_abinit_var_" in varname:
                dataset_vars[varname] = None

        dataset_vars.update(self.abinitVars[0])
        dataset_vars.update(self.abinitVars[self.current_dataset])

        # Take care of default values. We need to do this here because the default values of some variables depend on
        # the value of other variables.
        if dataset_vars["x_abinit_var_ntypat"] is None:
            dataset_vars["x_abinit_var_ntypat"] = 1
        if dataset_vars["x_abinit_var_npsp"] is None:
            dataset_vars["x_abinit_var_npsp"] = dataset_vars["x_abinit_var_ntypat"]
        if dataset_vars["x_abinit_var_nshiftk"] is None:
            dataset_vars["x_abinit_var_nshiftk"] = 1
        if dataset_vars["x_abinit_var_natrd"] is None:
            dataset_vars["x_abinit_var_natrd"] = dataset_vars["x_abinit_var_natom"]
        if dataset_vars["x_abinit_var_nsppol"] is None:
            dataset_vars["x_abinit_var_nsppol"] = 1
        if dataset_vars["x_abinit_var_nspden"] is None:
            dataset_vars["x_abinit_var_nspden"] = dataset_vars["x_abinit_var_nsppol"]
        if dataset_vars["x_abinit_var_nkpt"] is None:
            dataset_vars["x_abinit_var_nkpt"] = 1
        if dataset_vars["x_abinit_var_occopt"] is None:
            dataset_vars["x_abinit_var_occopt"] = 1
        if dataset_vars["x_abinit_var_ixc"] is None:
            dataset_vars["x_abinit_var_ixc"] = 1

        # Fix nband
        if len(dataset_vars["x_abinit_var_nband"].split()) == 1:
            nband = ""
            for ispin in range(int(dataset_vars["x_abinit_var_nsppol"])):
                for ikpt in range(int(dataset_vars["x_abinit_var_nkpt"])):
                    nband += dataset_vars["x_abinit_var_nband"]+" "
            dataset_vars["x_abinit_var_nband"] = nband

        for varname, varvalue in dataset_vars.items():

            meta_info = backend.superBackend.metaInfoEnv().infoKindEl(varname)

            # Skip optional variables that do not have a value or that are not defined in the meta-info
            if varvalue is None or meta_info is None:
                continue

            if varname == "x_abinit_var_occ":
                # Abinit allows for different numbers of bands per k-point and/or spin channel
                # This means the occupations need to be handled in a special way

                if dataset_vars["x_abinit_var_occopt"] != 2 and dataset_vars["x_abinit_var_nsppol"] == 1:
                    # In this case Abinit only prints the occupations for one k-point, as occupations are the same
                    # for all k-points
                    varvalue = ""
                    for ikpt in range(int(dataset_vars["x_abinit_var_nkpt"])):
                        varvalue += dataset_vars["x_abinit_var_occ"]+" "

                nband = sum([int(x) for x in dataset_vars["x_abinit_var_nband"].split()])
                array = np.array(varvalue.split(), dtype=parser_backend.numpyDtypeForDtypeStr(meta_info.dtypeStr))
                backend.addArrayValues(varname, array.reshape([nband]))
            elif varname == "x_abinit_var_ixc":
                # If no value of ixc is given in the input file, Abinit will try to choose it from the pseudopotentials.
                # Since the pseudopotentials are read while performing the calculations for a given dataset, ixc might
                # have been already read and stored. In that case we ignore the value stored in dataset_vars.
                if section["x_abinit_var_ixc"] is None:
                    backend.addValue(varname, backend.convertScalarStringValue(varname, varvalue))

            elif len(meta_info.shape) == 0:
                # This is a simple scalar
                backend.addValue(varname, backend.convertScalarStringValue(varname, varvalue))

            else:
                # This is an array
                array = np.array(varvalue.split(), dtype=parser_backend.numpyDtypeForDtypeStr(meta_info.dtypeStr))
                shape = []
                for dim in meta_info.shape:
                    if isinstance(dim, basestring):
                        # Replace all instances of Abinit variables that appear in the dimension
                        # with their actual values.
                        dim_regex = '(?P<abi_var>x_abinit_var_\w+)'
                        for mo in re.finditer(dim_regex, dim):
                            dim = re.sub(mo.group("abi_var"), str(dataset_vars[mo.group("abi_var")]), dim)
                        # In some cases the dimension is given as a numerical expression that needs to be evaluated
                        try:
                           dim = eval(dim)
                        except SyntaxError as e:
                           print(dim, meta_info.name, meta_info.shape)
                           raise e
                    shape.append(dim)
                backend.addArrayValues(varname, array.reshape(shape))

    def onClose_x_abinit_section_stress_tensor(self, backend, gIndex, section):
        """Trigger called when x_abinit_section_stress_tensor is closed.
        """
        xx = unit_conversion.convert_unit(section["x_abinit_stress_tensor_xx"][-1], "hartree/bohr^3")
        yy = unit_conversion.convert_unit(section["x_abinit_stress_tensor_yy"][-1], "hartree/bohr^3")
        zz = unit_conversion.convert_unit(section["x_abinit_stress_tensor_zz"][-1], "hartree/bohr^3")
        zy = unit_conversion.convert_unit(section["x_abinit_stress_tensor_zy"][-1], "hartree/bohr^3")
        zx = unit_conversion.convert_unit(section["x_abinit_stress_tensor_zx"][-1], "hartree/bohr^3")
        yx = unit_conversion.convert_unit(section["x_abinit_stress_tensor_yx"][-1], "hartree/bohr^3")
        backend.addArrayValues("stress_tensor", np.array([[xx, yx, zx], [yx, yy, zy], [zx, zy, zz]]))

    def onClose_x_abinit_section_var(self, backend, gIndex, section):
        """Trigger called when x_abinit_section_var is closed.
        """
        # We store all the variables read in a dictionary for latter use.
        m = re.search('[0-9]+$', section["x_abinit_varname"][0])
        if m is None:
            dataset = 0
        else:
            dataset = int(m.group(0))
            section["x_abinit_varname"][0] = re.sub('[0-9]+$', '', section["x_abinit_varname"][0])

        if dataset not in self.abinitVars.keys():
            self.abinitVars[dataset] = {}
        if len(section["x_abinit_varvalue"]) == 1:
            self.abinitVars[dataset]["x_abinit_var_" + section["x_abinit_varname"][0]] = section["x_abinit_varvalue"][0]
        else:
            # We have an array of values. We will only set the variable if we have all the values, that is, if the array
            # was not truncated.
            if section["x_abinit_vartruncation"] is None:
                self.abinitVars[dataset]["x_abinit_var_" + section["x_abinit_varname"][0]] = \
                    " ".join(section["x_abinit_varvalue"])


# description of the input
headerMatcher = \
    SM(name='Header',
       startReStr=r"\.Version\s*[0-9a-zA-Z_.]*\s*of ABINIT\s*$",
       required=True,
       forwardMatch=True,
       subMatchers=[SM(r"\.Version (?P<program_version>[0-9a-zA-Z_.]*) of ABINIT\s*$"),
                    SM(r"\.\((?P<x_abinit_parallel_compilation>[a-zA-Z]*)\s*version, prepared for a "
                       r"(?P<program_compilation_host>\S*)\s*computer\)\s*$"),
                    SM(r"\s*$"),
                    SM(startReStr="\.Copyright \(C\) 1998-[0-9]{4} ABINIT group .\s*$",
                       coverageIgnore=True,
                       subMatchers=[SM(r"\s*ABINIT comes with ABSOLUTELY NO WARRANTY.\s*$",
                                       coverageIgnore=True),
                                    SM(r"\s*It is free software, and you are welcome to redistribute it\s*$",
                                       coverageIgnore=True),
                                    SM(r"\s*under certain conditions \(GNU General Public License,\s*$",
                                       coverageIgnore=True),
                                    SM(r"\s*see ~abinit/COPYING or http://www.gnu.org/copyleft/gpl.txt\).\s*$",
                                       coverageIgnore=True),
                                    SM(r"\s*ABINIT is a project of the Universite Catholique de Louvain,\s*$",
                                       coverageIgnore=True),
                                    SM(r"\s*Corning Inc. and other collaborators, see "
                                       r"~abinit/doc/developers/contributors.txt .\s*$",
                                       coverageIgnore=True),
                                    SM(r"\s*Please read ~abinit/doc/users/acknowledgments.html for suggested\s*$",
                                       coverageIgnore=True),
                                    SM(r"\s*acknowledgments of the ABINIT effort.\s*$",
                                       coverageIgnore=True),
                                    SM(r"\s*For more information, see http://www.abinit.org .\s*$",
                                       coverageIgnore=True)
                                    ]
                       ),
                    SM(r"\.Starting date : (?P<x_abinit_start_date>[0-9a-zA-Z ]*)\.\s*$"),
                    SM(r"^- \( at\s*(?P<x_abinit_start_time>[0-9a-z]*)\s*\)\s*$"),
                    SM(r"^- input  file\s*->\s*(?P<x_abinit_input_file>\S*)\s*$"),
                    SM(r"^- output file\s*->\s*(?P<x_abinit_output_file>\S*)\s*$"),
                    SM(r"^- root for input  files\s*->\s*(?P<x_abinit_input_files_root>\S*)\s*$"),
                    SM(r"^- root for output files\s*->\s*(?P<x_abinit_output_files_root>\S*)\s*$")
                    ],
       )

timerMatcher = \
    SM(name='Timer',
       startReStr="- Total cpu\s*time",
       endReStr="={80}",
       required=True,
       forwardMatch=True,
       coverageIgnore=True,
       subMatchers=[SM(r"- Total cpu\s*time\s*\(\S*\):\s*(?P<x_abinit_total_cpu_time>[0-9.]+)\s*\S*\s*\S*\s*$"),
                    SM(r"- Total wall clock time\s*\(\S*\):\s*(?P<x_abinit_total_wallclock_time>[0-9.]+)\s*\S*\s*\S*\s*$"),
                    SM(r"-\s*$",
                       coverageIgnore=True),
                    SM(name="Profiling",
                       startReStr="- For major independent code sections, cpu and wall times \(sec\),",
                       endReStr="- subtotal(\s*[0-9.]+){4}",
                       repeats=True,
                       coverageIgnore=True,
                       subMatchers=[SM(r"-\s*as well as % of the (total time and number of calls|time and number of "
                                       r"calls for node [0-9]+-)",
                                       coverageIgnore=True),
                                    SM(r"-<BEGIN_TIMER mpi_nprocs = [0-9]+, omp_nthreads = [0-9]+, mpi_rank = "
                                       r"([0-9]+|world)>",
                                       coverageIgnore=True),
                                    SM(r"- cpu_time =\s*[0-9.]+, wall_time =\s*[0-9.]+",
                                       coverageIgnore=True),
                                    SM(r"-",
                                       coverageIgnore=True),
                                    SM(r"- routine\s*cpu\s*%\s*wall\s*%\s*number of calls\s*Gflops",
                                       coverageIgnore=True),
                                    SM(r"-\s*\(-1=no count\)",
                                       coverageIgnore=True),
                                    SM(r"-(\s*\S*)+(\s*[-0-9.]+){6}",
                                       coverageIgnore=True, repeats=True),
                                    SM(r"-<END_TIMER>",
                                       coverageIgnore=True),
                                    SM(r"-",
                                       coverageIgnore=True, required=False),
                                    ]
                       )
                    ]
       )

memestimationMatcher = \
    SM(name='MemEstimation',
       startReStr=r"\s*(Symmetries|DATASET\s*[0-9]{1,4})\s*:\s*space group\s*(\S*\s*){1,3}\s*\(\#\s*[0-9]{1,3}\);\s*"
                  r"Bravais\s*\S*\s*\([0-9a-zA-Z- .]*\)\s*$",
       endReStr=r"={80}",
       repeats=True,
       subMatchers=[SM(r"={80}",
                       coverageIgnore=True),
                    SM(r"\s*Values of the parameters that define the memory need (of the present run|for DATASET\s*"
                       r"[0-9]+\.)",
                       coverageIgnore=True),
                    # We ignore the variables printed here, as what is printed is Abinit version dependent and depends
                    # on the actual values of multiple parameters. The most important variables are repeated later.
                    SM(r"(-|P)?(\s*\S*\s*=\s*[0-9]+)+",
                       coverageIgnore=True, repeats=True),
                    SM(r"={80}",
                       coverageIgnore=True),
                    SM(r"P This job should need less than\s*[0-9.]+\s*Mbytes of memory.",
                       coverageIgnore=True),
                    SM(r"\s*Rough estimation \(10\% accuracy\) of disk space for files :",
                       coverageIgnore=True),
                    SM(r"_ WF disk file :\s*[0-9.]+\s*Mbytes ; DEN or POT disk file :\s*[0-9.]+\s*Mbytes.",
                       coverageIgnore=True),
                    SM(r"={80}",
                       coverageIgnore=True)
                    ]
       )

inputVarsMatcher = \
    SM(name='InputVars',
       startReStr=r"-{80}",
       endReStr=r"={80}",
       required=True,
       coverageIgnore=True,
       subMatchers=[SM(r"-{13} Echo of variables that govern the present computation -{12}",
                       coverageIgnore=True),
                    SM(r"-{80}",
                       coverageIgnore=True),
                    SM(r"-",
                       coverageIgnore=True),
                    SM(r"- outvars: echo of selected default values",
                       coverageIgnore=True),
                    SM(r"-(\s*\w+\s*=\s*[0-9]+\s*,{0,1})*"),
                    SM(r"-",
                       coverageIgnore=True),
                    SM(r"- outvars: echo of global parameters not present in the input file",
                       coverageIgnore=True),
                    SM(r"-(\s*\w+\s*=\s*[0-9]+\s*,{0,1})*"),
                    SM(r"-",
                       coverageIgnore=True),
                    SM(r" -outvars: echo values of preprocessed input variables --------",
                       coverageIgnore=True),
                    SM(startReStr=r"[-P]?\s+[_a-zA-Z0-9]+\s+(\S*\s*)+\s*(Hartree|Bohr)?\s*$",
                       forwardMatch=True,
                       sections=['x_abinit_section_var'],
                       repeats=True,
                       subMatchers=[SM(r"[-P]?\s+(?P<x_abinit_varname>[_a-zA-Z0-9]+)\s+"
                                       r"(?P<x_abinit_varvalue>((?!(Bohr|Hartree))\S*\s*)+)\s*(Hartree|Bohr)?\s*$"),
                                    SM(r"\s{20,}(?P<x_abinit_varvalue>([0-9Ee.+-]+\s*)+)\s*$",
                                       repeats=True),
                                    SM(r"\s{20,}outvar(_i_n|s)\s*: Printing only first\s*"
                                       r"(?P<x_abinit_vartruncation>[0-9]+)\s*[-a-zA-Z]*.\s*$",
                                       required=False)
                                    ]
                       ),
                    SM(r"={80}",
                       coverageIgnore=True),
                    SM(r"\s*chkinp: Checking input parameters for consistency(\.|,\s*jdtset=\s*[0-9]+\.)",
                       coverageIgnore=True, repeats=True)
                    ]
       )


eigenvaluesBlockMatcher = \
    SM(name="EigenvaluesBlock",
       startReStr=r"\s*Eigenvalues \(hartree\) for nkpt=\s*[0-9]+\s*k points(, SPIN (UP|DOWN))?:\s*$",
       repeats=True,
       subMatchers=[SM(startReStr=r"\s*kpt#\s*[0-9]+, nband=\s*[0-9]+, wtk=\s*[0-9.]+\s*, kpt=(\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?){3}"
                                  r"\s*\(reduced coord\)\s*$",
                       forwardMatch=True,
                       repeats=True,
                       subMatchers=[SM(r"\s*kpt#\s*[0-9]+, nband=\s*[0-9]+, wtk=\s*(?P<x_abinit_wtk>[0-9.]+)\s*, "
                                       r"kpt=(?P<x_abinit_kpt>(\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?){3})\s*\(reduced coord\)\s*$"),
                                    SM(r"(?P<x_abinit_eigenvalues>(\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)+)\s*$",
                                       repeats=True),
                                    SM(r"\s*occupation numbers for kpt#\s+[0-9]+\s*$",
                                       required=False),
                                    SM(r"(?P<x_abinit_occupations>(\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)+)\s*$",
                                       repeats=True, required=False)
                                    ]
                       ),
                    ]
       )

SCFResultsMatcher = \
    SM(name='SCFResults',
       startReStr=r"\s*----iterations are completed or convergence reached----\s*$",
       required=False,
       subMatchers=[SM(r"\s*Mean square residual over all n,k,spin=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*;\s*max=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*$",
                       coverageIgnore=True),
                    SM(startReStr=r"\s*([-+0-9.]+\s*){3}\s*[0-9]?\s*[-+0-9eEdD.]+\s*kpt; spin; max resid\(k\); each band:\s*$",
                       coverageIgnore=True,
                       repeats=True,
                       subMatchers=[SM(r"\s*([-+0-9eEdD.]+\s*)+\s*$",
                                       coverageIgnore=True)
                                    ]
                       ),
                    SM(startReStr=r"\s*reduced coordinates \(array xred\) for\s*[0-9]+\s*atoms\s*$",
                       coverageIgnore=True,
                       subMatchers=[SM(r"\s*([-+0-9.]+\s*){3}\s*$",
                                       coverageIgnore=True, repeats=True)
                                    ]
                       ),
                    SM(startReStr=r"\s*rms dE/dt=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*; max dE/dt=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*; dE/dt below "
                                  r"\(all hartree\)\s*$",
                       coverageIgnore=True,
                       subMatchers=[SM(r"\s*[0-9]+\s*([-+0-9.]+\s*){3}\s*$",
                                       coverageIgnore=True, repeats=True)
                                    ]
                       ),
                    SM(startReStr=r"\s*cartesian coordinates \(angstrom\) at end:\s*$",
                       subMatchers=[SM(r"\s*[0-9]+(?P<x_abinit_atom_xcart_final>(\s*[-+0-9.]+){3})\s*$",
                                       repeats=True)
                                    ]
                       ),
                    SM(startReStr=r"\s*cartesian forces \(hartree/bohr\) at end:\s*$",
                       subMatchers=[SM(r"\s*[0-9]+(?P<x_abinit_atom_force_final>(\s*[-+0-9.]+){3})\s*$",
                                       repeats=True),
                                    SM(r"\s*frms,max,avg=(\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?){5}\s*h/b\s*$")
                                    ]
                       ),
                    SM(startReStr=r"\s*cartesian forces \(eV/Angstrom\) at end:\s*$",
                       subMatchers=[SM(r"\s*[0-9]+(\s*[-+0-9.]+){3}\s*$",
                                       repeats=True),
                                    SM(r"\s*frms,max,avg=(\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?){5}\s*e/A\s*$")
                                    ]
                       ),
                    SM(r"\s*length scales=(\s*[0-9.]+){3}\s*bohr\s*$",
                       coverageIgnore=True),
                    SM(r"\s*=(\s*[0-9.]+){3}\s*angstroms\s*$",
                       coverageIgnore=True),
                    SM(r"\s*prteigrs : about to open file\s*(?P<x_abinit_eig_filename>\S*)\s*$",
                       required=False),
                    SM(r"\s*Fermi \(or HOMO\) energy \(hartree\) =\s*(?P<x_abinit_fermi_energy>[-+0-9.]+)\s*"
                       r"Average Vxc \(hartree\)=\s*[-+0-9.]+\s*$"),
                    SM(r"\s*Magnetisation \(Bohr magneton\)=\s*(?P<x_abinit_magnetisation>[-+0-9.eEdD]*)\s*$"),
                    SM(r"\s*Total spin up =\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*Total spin down =\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*$",
                       coverageIgnore=True),
                    SM(name="Eigenvalues",
                       startReStr=r"\s*Eigenvalues \(hartree\) for nkpt=\s*[0-9]+\s*k points(, SPIN (UP|DOWN))?:\s*$",
                       forwardMatch=True,
                       sections=["section_eigenvalues"],
                       subMatchers=[eigenvaluesBlockMatcher]
                       ),
                    SM(startReStr=r"\s*(Total charge density|Spin up density|Spin down density|"
                                  r"Magnetization \(spin up - spin down\)|"
                                  r"Relative magnetization \(=zeta, between -1 and 1\))\s*(\[el/Bohr\^3\])?\s*$",
                       coverageIgnore=True,
                       repeats=True,
                       subMatchers=[SM(r",(Next)?\s*(m|M)(axi|ini)mum=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*at reduced coord."
                                       r"(\s*[0-9.]+){3}\s*$",
                                       repeats=True),
                                    SM(r",\s*Integrated=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*$"),
                                    ]
                       ),
                    SM(r"-{80}",
                       coverageIgnore=True),
                    SM(r"\s*Components of total free energy \(in Hartree\) :\s*$",
                       coverageIgnore=True),
                    SM(r"\s*Kinetic energy\s*=\s*(?P<x_abinit_energy_kinetic>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*$"),
                    SM(r"\s*Hartree energy\s*=\s*(?P<x_abinit_energy_hartree>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*$"),
                    SM(r"\s*XC energy\s*=\s*(?P<x_abinit_energy_xc>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*$"),
                    SM(r"\s*Ewald energy\s*=\s*(?P<x_abinit_energy_ewald>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*$"),
                    SM(r"\s*PspCore energy\s*=\s*(?P<x_abinit_energy_psp_core>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*$"),
                    SM(r"\s*Loc. psp. energy\s*=\s*(?P<x_abinit_energy_psp_local>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*$"),
                    SM(r"\s*NL   psp  energy\s*=\s*(?P<x_abinit_energy_psp_nonlocal>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*$"),
                    SM(r"\s*>{5}\s*Internal E=\s*(?P<x_abinit_energy_internal>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*$"),
                    SM(r"\s*-kT\*entropy\s*=\s*(?P<x_abinit_energy_ktentropy>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*$"),
                    SM(r"\s*>{9}\s*Etotal=\s*(?P<x_abinit_energy_total>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*$"),
                    SM(r"\s*Other information on the energy :\s*$",
                       coverageIgnore=True),
                    SM(r"\s*Total energy\(eV\)=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*;\s*Band energy \(Ha\)=\s*"
                       r"(?P<x_abinit_energy_band>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*$"),
                    SM(r"-{80}",
                       coverageIgnore=True),
                    SM(startReStr=r"\s*rms coord change=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*atom, delta coord \(reduced\):\s*$",
                       subMatchers=[SM(r"\s*\d+(\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?){3}\s*$",
                                       repeats=True),
                                    ]),
                    SM(r"\s*Cartesian components of stress tensor \(hartree/bohr\^3\)\s*$",
                       coverageIgnore=True),
                    SM(r"\s*sigma\(1 1\)=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*sigma\(3 2\)=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*$",
                       coverageIgnore=True),
                    SM(r"\s*sigma\(2 2\)=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*sigma\(3 1\)=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*$",
                       coverageIgnore=True),
                    SM(r"\s*sigma\(3 3\)=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*sigma\(2 1\)=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*$",
                       coverageIgnore=True),
                    SM(r"-Cartesian components of stress tensor \(GPa\)\s*\[Pressure=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*GPa]\s*$",
                       coverageIgnore=True),
                    SM(r"- sigma\(1 1\)=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*sigma\(3 2\)=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*$",
                       coverageIgnore=True),
                    SM(r"- sigma\(2 2\)=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*sigma\(3 1\)=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*$",
                       coverageIgnore=True),
                    SM(r"- sigma\(3 3\)=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*sigma\(2 1\)=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*$",
                       coverageIgnore=True)
                    ]
       )

SCFOutput = \
    SM(name='SCFOutput',
       startReStr=r"-{3}OUTPUT-{71}\s*$",
       required=False,
       subMatchers=[SM(startReStr=r"\s*Cartesian coordinates \(xcart\) \[bohr\]\s*$",
                       subMatchers=[SM(r"\s*(?P<x_abinit_atom_xcart>(\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?){3})\s*$",
                                       repeats=True)]
                       ),
                    SM(startReStr=r"\s*Reduced coordinates \(xred\)\s*$",
                       subMatchers=[SM(r"\s*(\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?){3}\s*$",
                                       repeats=True)]
                       ),
                    SM(startReStr=r"\s*Cartesian forces \(fcart\) \[Ha/bohr\]; max,rms=(\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?){2}\s*"
                                  r"\(free atoms\)\s*$",
                       subMatchers=[SM(r"\s*(?P<x_abinit_atom_force>(\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?){3})\s*$",
                                       repeats=True)]
                       ),
                    SM(startReStr=r"\s*Reduced forces \(fred\)\s*",
                       subMatchers=[SM(r"\s*(\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?){3}\s*$",
                                       repeats=True)]
                       ),
                    SM(r"\s*Total energy \(etotal\) \[Ha\]=\s*(?P<energy_total__hartree>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*$"),
                    SM(r"\s*Difference of energy with previous step \(new-old\):\s*$",
                       coverageIgnore=True),
                    SM(r"\s*Absolute \(Ha\)=[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*$"),
                    SM(r"\s*Relative\s*=[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*$"),
                    SM(r"fconv : WARNING -\s*$",
                       coverageIgnore=True),
                    SM(r"\s*(At Broyd/MD step\s*\d+\s*, gradients|ntime=\s*\d+\s*was|fconv : at Broyd/MD step\s*\d+\s*, gradients have) "
                       r"(?P<x_abinit_geometry_optimization_converged>(are converged|not enough|not converged)) "
                       r"(:|Broyd/MD steps to converge gradients:|yet.)\s*$"),
                    SM(r"\s*max grad \(force/stress\) =\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*< tolmxf=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*ha/bohr \(free atoms\)\s*")
                    ]
       )


SCFCycleMatcher = \
    SM(name='SCFCycle',
       startReStr=r"(\s*iter\s*(Etot\(hartree\)|2DEtotal\(Ha\))\s*deltaE\((h|Ha)\)(\s*\w+)*|"
                  r"--- Iteration: \(\s*\d+/\d+\) Internal Cycle: \(\d+/\d+\))\s*$",
       repeats=True,
       sections=['section_single_configuration_calculation'],
       subMatchers=[SM(r"-{80}\s*$",
                       coverageIgnore=True),
                    SM(r"---SELF-CONSISTENT-FIELD CONVERGENCE-{44}",
                       coverageIgnore=True),
                    SM(r"(\s*|-)ETOT\s*[0-9]+\s*(?P<energy_total_scf_iteration__hartree>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*"
                       r"(?P<energy_change_scf_iteration__hartree>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)(\s*[-+0-9.eEdD]*)*",
                       sections=["section_scf_iteration"],
                       repeats=True),
                    SM(r"\s*At SCF step\s*(?P<number_of_scf_iterations>[0-9]+)\s*"
                       r"(, etot|, forces|vres2\s*=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*<\s*tolvrs=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*=>)\s*"
                       r"(?P<x_abinit_single_configuration_calculation_converged>(is converged|are converged|converged))"
                       r"\s*(:|.)\s*$"),
                    SM(r"\s*for the second time, (max diff in force|diff in etot)=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*<\s*tol(dfe|dff)="
                       r"\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*$"),
                    SM(startReStr=r"\s*Cartesian components of stress tensor \(hartree/bohr\^3\)\s*$",
                       coverageIgnore=True,
                       sections=["x_abinit_section_stress_tensor"],
                       subMatchers=[SM(r"\s*sigma\(1 1\)=\s*(?P<x_abinit_stress_tensor_xx>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"
                                       r"\s*sigma\(3 2\)=\s*(?P<x_abinit_stress_tensor_zy>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*$"),
                                    SM(r"\s*sigma\(2 2\)=\s*(?P<x_abinit_stress_tensor_yy>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"
                                       r"\s*sigma\(3 1\)=\s*(?P<x_abinit_stress_tensor_zx>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*$"),
                                    SM(r"\s*sigma\(3 3\)=\s*(?P<x_abinit_stress_tensor_zz>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"
                                       r"\s*sigma\(2 1\)=\s*(?P<x_abinit_stress_tensor_yx>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*$")
                                    ]
                       ),
                    SM(startReStr=r"\s*Integrated electronic density in atomic spheres:\s*$",
                       required=False,
                       coverageIgnore=True,
                       subMatchers=[SM(r"\s*-{48}\s*$",
                                       coverageIgnore=True),
                                    SM(r"\s*Atom\s*Sphere_radius\s*Integrated_density\s*$",
                                       coverageIgnore=True),
                                    SM(r"\s*\d+\s*[0-9.]+\s*[0-9.]+\s*$",
                                       coverageIgnore=True, repeats=True)
                                    ]
                       ),
                    SM(startReStr=r"\s*Integrated electronic and magnetization densities in atomic spheres:\s*$",
                       required=False,
                       coverageIgnore=True,
                       subMatchers=[SM(r"\s*-{69}\s*$",
                                       coverageIgnore=True),
                                    SM(r"\s*Note: Diff\(up-dn\) is a rough approximation of local magnetic moment\s*$",
                                       coverageIgnore=True),
                                    SM(r"\s*Atom\s*Radius\s*up_density\s*dn_density\s*Total\(up\+dn\)\s*Diff\(up-dn\)\s*$",
                                       coverageIgnore=True),
                                    SM(r"\s*\d+(\s*[0-9.]+){5}\s*$",
                                       coverageIgnore=True, repeats=True),
                                    SM(r"\s*-{69}\s*$",
                                       coverageIgnore=True),
                                    SM(r"\s*Sum:(\s*[0-9.]+){4}\s*$",
                                       coverageIgnore=True),
                                    SM(r"\s*Total magnetization \(from the atomic spheres\):\s*[0-9.]+\s*$",
                                       coverageIgnore=True),
                                    SM(r"\s*Total magnetization \(exact up - dn\):\s*[0-9.]+\s*$",
                                       coverageIgnore=True)
                                    ]
                       ),
                    SCFOutput,
                    SM(r"={80}\s*$",
                       coverageIgnore=True, required=False),
                    SCFResultsMatcher
                    ]
       )


pseudopotentialMatcher = \
    SM(name="pseudopotential",
       startReStr=r"-\s*pspini: atom type\s*[0-9]+\s*psp file is \S*\s*$",
       endReStr=r"\s*pspatm: atomic psp has been read  and splines computed\s*$",
       forwardMatch=True,
       repeats=True,
       coverageIgnore=True,
       subMatchers=[SM(r"-\s*pspini: atom type\s*\d+\s*psp file is\s*\S*\s*$"),
                    SM(r"-\s*pspatm: opening atomic psp file\s*\S*",
                       coverageIgnore=True),
                    SM(r"-(\s*[0-9.]+){3}\s*znucl, zion, pspdat\s*$"),
                    SM(r"(\s*\d+){5}\s*[0-9.]+\s*pspcod,pspxc,lmax,lloc,mmax,r2well\s*$"),
                    SM(startReStr=r"\s*\d+\s*[0-9.]+\s*[0-9.]+\s*\d+\s*[0-9.]+\s*l,e99.0,e99.9,nproj,rcpsp\s*$",
                       repeats=True,
                       subMatchers=[SM(r"\s*([0-9.]+\s*){4}rms, ekb1, ekb2, epsatm\s*$")]),
                    SM(r"\s*([0-9.]+\s*){3}rchrg,fchrg,qchrg\s*$"),
                    SM(startReStr=r"\s*rloc=\s*[0-9.]+\s*$",
                       subMatchers=[SM(r"\s*cc1=\s*[-0-9.]+; cc2=\s*[-0-9.]+; cc3=\s*[-0-9.]+; cc4=\s*[-0-9.]+\s*$"),
                                    SM(r"\s*rrs=\s*[-0-9.]+; h1s=\s*[-0-9.]+; h2s=\s*[-0-9.]+\s*$"),
                                    SM(r"\s*rrp=\s*[-0-9.]+; h1p=\s*[-0-9.]+\s*$"),
                                    SM(r"-  Local part computed in reciprocal space.\s*$")
                                    ]
                       ),
                    SM(r"\s*pspatm : COMMENT -\s*$",
                       coverageIgnore=True),
                    SM(r"\s*the projectors are not normalized,\s*",
                       coverageIgnore=True),
                    SM(r"\s*so that the KB energies are not consistent with\s*$",
                       coverageIgnore=True),
                    SM(r"\s*definition in PRB44, 8503 \(1991\).\s*$",
                       coverageIgnore=True),
                    SM(r"\s*However, this does not influence the results obtained hereafter.\s*$",
                       coverageIgnore=True),
                    SM(r"\s*pspatm: epsatm=\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*$"),
                    SM(r"\s*--- l  ekb\(1:nproj\) -->\s*$",
                       coverageIgnore=True),
                    SM(r"\s*\d+\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*$",
                       repeats=True)
                    ]
       )

datasetHeaderMatcher = \
    SM(name='DatasetHeader',
       startReStr=r"={2}\s*DATASET\s*[0-9]+\s*={66}\s*$",
       endReStr=r"={80}\s*$",
       forwardMatch=True,
       repeats=False,
       sections=['x_abinit_section_dataset_header'],
       subMatchers=[SM(r"={2}\s*DATASET\s*(?P<x_abinit_dataset_number>[0-9]+)\s*={66}\s*$"),
                    SM(r"-\s*nproc\s*=\s*[0-9]+\s*$"),
                    SM(name="XC",
                       startReStr=r"\s*Exchange-correlation functional for the present dataset will be:",
                       required=False,
                       coverageIgnore=True,
                       subMatchers=[SM(r"(\s*\S*)+\s*-\s*ixc=(?P<x_abinit_var_ixc>[-0-9]+)\s*$"),
                                    SM(r"\s*Citation for XC functional:",
                                       coverageIgnore=True),
                                    SM(r"\s*((\S*.\s*)+\S*,)+\s*\S*\s*[0-9]+,\s*[0-9]+\s*\([0-9]+\)\s*$",
                                       coverageIgnore=True)
                                    ]
                       ),
                    SM(r"\s*Real\(R\)\+Recip\(G\) space primitive vectors, cartesian coordinates \(Bohr,Bohr\^-1\):",
                       coverageIgnore=True),
                    SM(r"\s*R\(1\)=(?P<x_abinit_vprim_1>(\s*[0-9.-]+){3})\s*G\(1\)=(\s*[0-9.-]+){3}\s*$"),
                    SM(r"\s*R\(2\)=(?P<x_abinit_vprim_2>(\s*[0-9.-]+){3})\s*G\(2\)=(\s*[0-9.-]+){3}\s*$"),
                    SM(r"\s*R\(3\)=(?P<x_abinit_vprim_3>(\s*[0-9.-]+){3})\s*G\(3\)=(\s*[0-9.-]+){3}\s*$"),
                    SM(r"\s*Unit cell volume ucvol=\s*[-+0-9.eEdD]*\s*bohr\^3\s*$"),
                    SM(r"\s*Angles \(23,13,12\)=(\s*[-+0-9.eEdD]*){3}\s*degrees\s*$"),
                    SM(r"\s*getcut: wavevector=(\s*[0-9.]*){3}\s*ngfft=(\s*[0-9]*){3}\s*$"),
                    SM(r"\s*ecut\(hartree\)=\s*[0-9.]*\s*=> boxcut\(ratio\)=\s*[0-9.]*\s*$"),
                    SM(r"--- Pseudopotential description ------------------------------------------------",
                       coverageIgnore=True),
                    pseudopotentialMatcher,
                    SM(r"\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*ecore\*ucvol\(ha\*bohr\*\*3\)\s*$",
                       coverageIgnore=True),
                    SM(r"-{80}",
                       coverageIgnore=True),
                    SM(r"P newkpt: treating\s*[0-9]+\s*bands with npw=\s*[0-9]+\s*for ikpt=\s*[0-9]+\s*by node\s*[0-9]+",
                       coverageIgnore=True,
                       repeats=True,
                       required=False),
                    SM(r"_setup2: Arith. and geom. avg. npw \(full set\) are(\s*[0-9.]+\s*){2}",
                       coverageIgnore=True)
                    ]
       )

datasetMatcher = \
    SM(name='Dataset',
       startReStr=r"={2}\s*DATASET\s*[0-9]+\s*={66}\s*$",
       forwardMatch=True,
       repeats=True,
       sections=['x_abinit_section_dataset'],
       subMatchers=[datasetHeaderMatcher,
                    SM("===\s*\[ionmov=\s*\d+\]\s*\S*\s*method\s*\(forces(,Tot energy){0,1}\)\s*$",
                       required=False, coverageIgnore=True),
                    SM(r"={80}\s*$",
                       coverageIgnore=True, required=False, weak=True),
                    SCFCycleMatcher,
                    SM(r"==( END DATASET\(S\) |={16})={62}\s*$",
                       coverageIgnore=True),
                    SM(r"={80}\s*$",
                       coverageIgnore=True, required=False, weak=True)
                    ]
       )


outputVarsMatcher = \
    SM(name='OutputVars',
       startReStr=r"\s*-outvars: echo values of variables after computation  --------",
       endReStr=r"={80}",
       coverageIgnore=True,
       required=True,
       subMatchers=[SM(startReStr=r"[-P]?\s+[_a-zA-Z0-9]+\s+(\S*\s*)+\s*(Hartree|Bohr)?\s*$",
                       repeats=True,
                       coverageIgnore=True,
                       subMatchers=[SM(r"\s{20,}(\S*\s*)+\s*$",
                                       repeats=True, coverageIgnore=True),
                                    SM(r"\s{20,}outvar(_i_n|s)\s*: Printing only first\s*[0-9]*\s*[-a-zA-Z]*.\s*$",
                                       required=False, coverageIgnore=True)
                                    ]
                       )
                    ]
       )

footerMatcher = \
    SM(name='Footer',
       startReStr="\s*Suggested references for the acknowledgment of ABINIT usage.\s*",
       required=True,
       coverageIgnore=True,
       subMatchers=[SM(r"\s*The users of ABINIT have little formal obligations with respect to the ABINIT group",
                       coverageIgnore=True),
                    SM(r"\s*\(those specified in the GNU General Public License, "
                       r"http://www.gnu.org/copyleft/gpl.txt\).",
                       coverageIgnore=True),
                    SM(r"\s*However, it is common practice in the scientific literature,",
                       coverageIgnore=True),
                    SM(r"\s*to acknowledge the efforts of people that have made the research possible.",
                       coverageIgnore=True),
                    SM(r"\s*In this spirit, please find below suggested citations of work written by ABINIT "
                       r"developers,",
                       coverageIgnore=True),
                    SM(r"\s*corresponding to implementations inside of ABINIT that you have used in the present run.",
                       coverageIgnore=True),
                    SM(r"\s*Note also that it will be of great value to readers of publications presenting these "
                       r"results,",
                       coverageIgnore=True),
                    SM(r"\s*to read papers enabling them to understand the theoretical formalism and details",
                       coverageIgnore=True),
                    SM(r"\s*of the ABINIT implementation.",
                       coverageIgnore=True),
                    SM(r"\s*For information on why they are suggested, see also "
                       r"http://www.abinit.org/about/\?text=acknowledgments.",
                       coverageIgnore=True),
                    SM(r"-?(\s*\[[0-9]+\])?(\s*\S*)*",
                       coverageIgnore=True, weak=True, repeats=True),
                    SM(r"- Proc\.\s*[0-9]+\s*individual time \(sec\): cpu=\s*[0-9.]+\s*wall=\s*[0-9.]+\s*",
                       coverageIgnore=True),
                    SM(r"={80}",
                       coverageIgnore=True),
                    SM(r"\s*(?P<x_abinit_completed>Calculation completed)."),
                    SM(r".Delivered\s*[0-9]+\s*WARNINGs and\s*[0-9]+\s*COMMENTs to log file.",
                       coverageIgnore=True),
                    SM(r"\+Overall time at end \(sec\) : cpu=\s*[0-9.]+\s*wall=\s*[0-9.]+",
                       coverageIgnore=True)
                    ]
       )


mainFileDescription = \
    SM(name='root',
       startReStr="\s*$",
       required=True,
       subMatchers=[SM(name='NewRun',
                       startReStr="\s*$",
                       endReStr=r"\s*Overall time at end \(sec\) : cpu=\s*\S*\s*wall=\s*\S*",
                       required=True,
                       fixedStartValues={'program_name': 'ABINIT', 'program_basis_set_type': 'plane waves'},
                       sections=['section_run'],
                       subMatchers=[headerMatcher,
                                    memestimationMatcher,
                                    inputVarsMatcher,
                                    SM(r"={80}", coverageIgnore=True),
                                    datasetMatcher,
                                    outputVarsMatcher,
                                    timerMatcher,
                                    footerMatcher
                                    ]
                       )
                    ]
       )

class AbinitParser():
   """ A proper class envolop for running this parser from within python. """
   def __init__(self, backend, **kwargs):
       self.backend_factory = backend

   def parse(self, mainfile):
       from unittest.mock import patch
       logging.info('abinit parser started')
       logging.getLogger('nomadcore').setLevel(logging.WARNING)
       backend = self.backend_factory("abinit.nomadmetainfo.json")
       with patch.object(sys, 'argv', ['<exe>', '--uri', 'nmd://uri', mainfile]):
           mainFunction(
               mainFileDescription,
               None,
               parserInfo,
               cachingLevelForMetaName = {'x_abinit_section_var': CachingLevel.Cache
                                         },
               superContext=ABINITContext(),
               superBackend=backend)

       return backend
