import setup_paths
from builtins import object
from nomadcore.caching_backend import CachingLevel
from nomadcore.simple_parser import SimpleMatcher as SM
from nomadcore.simple_parser import mainFunction
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
from nomadcore import parser_backend
from nomadcore.unit_conversion import unit_conversion
from ase.data import chemical_symbols
from AbinitXC import ABINIT_NATIVE_IXC, ABINIT_LIBXC_IXC
import numpy as np
import re
import os
import logging
import time

logger = logging.getLogger("nomad.ABINITParser")

parserInfo = {
  "name": "ABINIT_parser",
  "version": "1.0"
}

# loading metadata from nomad-meta-info/meta_info/nomad_meta_info/abinit.nomadmetainfo.json
metaInfoPath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../../../../nomad-meta-info/meta_info/nomad_meta_info/abinit.nomadmetainfo.json"))
metaInfoEnv, warnings = loadJsonFile(filePath=metaInfoPath,
                                     dependencyLoader=None,
                                     extraArgsHandling=InfoKindEl.ADD_EXTRA_ARGS,
                                     uri=None)


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

    def onClose_section_method(self, backend, gIndex, section):
        """Trigger called when section_method is closed.
        """
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
            atom_types.append(symbol+str(species_count[symbol]))
        atom_labels = backend.arrayForMetaInfo("atom_labels", self.input["x_abinit_var_natom"][-1])
        for atom_index in range(self.input["x_abinit_var_natom"][-1]):
            atom_labels[atom_index] = atom_types[self.input["x_abinit_var_typat"][-1][atom_index] - 1]
        backend.addArrayValues("atom_labels", atom_labels)

        if self.input["x_abinit_var_xcart"] is None:
            if self.input["x_abinit_var_natom"][0] == 1:
                backend.addArrayValues("atom_positions", np.array([[0, 0, 0]]))
            else:
                logger.error("Positions of atoms is not available")
        else:
            backend.addArrayValues("atom_positions", self.input["x_abinit_var_xcart"][-1])

        backend.addArrayValues("configuration_periodic_dimensions", np.array([True, True, True]))

        backend.addValue("number_of_atoms", self.input["x_abinit_var_natom"][-1])

        backend.addValue("spacegroup_3D_number", self.input["x_abinit_var_spgroup"][-1])

        vprim = backend.arrayForMetaInfo("simulation_cell", [3, 3])
        for axis in [1, 2, 3]:
            if section["x_abinit_vprim_%s"%axis] is None:
                vprim[axis-1] = self.initial_simulation_cell[axis-1]
            else:
                vprim[axis-1] = section["x_abinit_vprim_%s"%axis][-1].split()
            for component in range(3):
                vprim[axis-1][component] = unit_conversion.convert_unit(vprim[axis-1][component], 'bohr')
        backend.addArrayValues("simulation_cell", vprim)

    def onOpen_x_abinit_section_dataset_header(self, backend, gIndex, section):
        """Trigger called when x_abinit_section_dataset is opened.
        """
        self.inputGIndex = backend.openSection("x_abinit_section_input")

    def onClose_x_abinit_section_dataset_header(self, backend, gIndex, section):
        """Trigger called when x_abinit_section_dataset is closed.
        """
        self.current_dataset = section["x_abinit_dataset_number"][-1]
        backend.closeSection("x_abinit_section_input", self.inputGIndex)
        self.methodGIndex = backend.openSection("section_method")
        backend.closeSection("section_method", self.methodGIndex)

        self.initial_simulation_cell = []
        for axis in [1, 2, 3]:
            self.initial_simulation_cell.append(section["x_abinit_vprim_%s" % axis][-1].split())

    def onOpen_section_single_configuration_calculation(self, backend, gIndex, section):
        """Trigger called when section_single_configuration_calculation is opened.
        """
        self.systemGIndex = backend.openSection("section_system")
        backend.addValue("single_configuration_calculation_to_system_ref", self.systemGIndex)
        backend.addValue("single_configuration_to_calculation_method_ref", self.methodGIndex)

    def onClose_section_single_configuration_calculation(self, backend, gIndex, section):
        """Trigger called when section_single_configuration_calculation is closed.
        """
        backend.closeSection("section_system", self.systemGIndex)

        if section["x_abinit_energy_xc"] is not None:
            backend.addValue("energy_XC", unit_conversion.convert_unit(section["x_abinit_energy_xc"][-1], "hartree"))
        if section["x_abinit_energy_kinetic"] is not None:
            backend.addValue("electronic_kinetic_energy",
                             unit_conversion.convert_unit(section["x_abinit_energy_kinetic"][-1], "hartree"))


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
                    atom_forces[n_atom, dir] = unit_conversion.convert_unit(float(force_string.split()[dir]), "forceAu")
                n_atom += 1
            backend.addArrayValues("atom_forces_raw", atom_forces)

    def onOpen_x_abinit_section_input(self, backend, gIndex, section):
        """Trigger called when x_abinit_section_input is opened.
        """
        self.input = section

    def onClose_x_abinit_section_input(self, backend, gIndex, section):
        """Trigger called when x_abinit_section_input is closed.
        """
        dataset_vars = {}
        for varname in metaInfoEnv.infoKinds.keys():
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

            meta_info = metaInfoEnv.infoKindEl(varname)

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
                    backend.addValue(varname, dataset_vars["x_abinit_var_ixc"])

            elif len(meta_info.shape) == 0:
                # This is a simple scalar
                backend.addValue(varname, backend.convertScalarStringValue(varname, varvalue))

            else:
                # This is an array
                array = np.array(varvalue.split(), dtype=parser_backend.numpyDtypeForDtypeStr(meta_info.dtypeStr))
                shape = []
                for dim in meta_info.shape:
                    if isinstance(dim, str):
                        # Replace all instances of Abinit variables that appear in the dimension
                        # with their actual values.
                        dim_regex = '(?P<abi_var>x_abinit_var_\w+)'
                        for mo in re.finditer(dim_regex, dim):
                            dim = re.sub(mo.group("abi_var"), str(dataset_vars[mo.group("abi_var")]), dim)
                        # In some cases the dimension is given as a numerical expression that needs to be evaluated
                        dim = eval(dim)
                    shape.append(dim)
                backend.addArrayValues(varname, array.reshape(shape))

    def onClose_x_abinit_section_var(self, backend, gIndex, section):
        """Trigger called when x_abinit_section_var is closed.
        """
        # We store all the variables read in a dictionary for latter use.
        if section["x_abinit_vardtset"] is None:
            dataset = 0
        else:
            dataset = section["x_abinit_vardtset"][0]
        if dataset not in self.abinitVars.keys():
            self.abinitVars[dataset] = {}
        if len(section["x_abinit_varvalue"]) == 1:
            self.abinitVars[dataset]["x_abinit_var_" + section["x_abinit_varname"][0]] = section["x_abinit_varvalue"][0]
        else:
            self.abinitVars[dataset]["x_abinit_var_" + section["x_abinit_varname"][0]] = \
                " ".join(section["x_abinit_varvalue"])


def build_abinit_vars_submatcher(is_output=False):
    matchers = []

    # Generate a dict of Abinit variables with the corresponding regex pattern.
    abi_vars = {}
    for varname in metaInfoEnv.infoKinds.keys():
        if "x_abinit_var_" in varname:
            meta_info = metaInfoEnv.infoKindEl(varname)
            if meta_info.dtypeStr == "f":
                pattern = "[-+0-9.eEdD]+"
            elif meta_info.dtypeStr == "i":
                pattern = "[-+0-9]+"
            elif meta_info.dtypeStr == "C":
                pattern = "[\w+]"
            else:
                raise Exception("Data type not supported")
            abi_vars[re.sub("x_abinit_var_", "", varname)] = pattern

    # Some variables that Abinit writes to the output are not documented as input variables so we add them here to the
    # dictionary.
    abi_vars.update({"mkmem": "[-+0-9]+"})
    if is_output:
        abi_vars.update(dict(etotal="[-+0-9.eEdD]+", fcart="[-+0-9.eEdD]+", strten="[-+0-9.eEdD]+"))

    # Currently we cannot create matchers for all the Abinit input variables as this would generate more than 100 named
    # groups in regexp. Therefore we will only try to parse a subset of the input variables. This should be changed once
    # this problem is fixed.
    supported_vars = \
        ["acell", "amu", "bs_loband", "diemac", "ecut", "etotal", "fcart", "fftalg", "ionmov", "iscf", "istwfk", "ixc",
         "jdtset", "natom", "kpt", "kptopt", "kptrlatt", "kptrlen", "mkmem", "nband", "ndtset", "ngfft", "nkpt",
         "nspden", "nsppol", "nstep", "nsym", "ntime", "ntypat", "occ", "occopt", "optforces", "prtdos", "rprim",
         "shiftk", "spgroup", "spinat", "strten", "symafm", "symrel", "tnons", "toldfe", "toldff", "tolmxf", "tolvrs",
         "tsmear", "typat", "wtk", "xangst", "xcart", "xred", "znucl"]
    for varname in sorted(abi_vars):
        if varname in supported_vars:
            matchers.append(SM(startReStr=r"[-P]?\s+%s[0-9]{0,4}\s+(%s\s*)+\s*(Hartree|Bohr)?\s*$"
                                          % (varname, abi_vars[varname]),
                               forwardMatch=True,
                               sections=['x_abinit_section_var'],
                               repeats=True,
                               subMatchers=[SM(r"[-P]?\s+(?P<x_abinit_varname>%s)((?P<x_abinit_vardtset>[0-9]{1,4})\s+|"
                                               r"\s+)(?P<x_abinit_varvalue>(%s\s*)+)\s*(Hartree|Bohr)?\s*$"
                                               % (varname, abi_vars[varname])),
                                            SM(r"\s{20,}(?P<x_abinit_varvalue>(%s\s*)+)\s*$" % (abi_vars[varname]),
                                               repeats=True)
                                            ]
                               )
                            )
    matchers.append(SM(r"\s*", coverageIgnore=True))
    return matchers

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
       startReStr=r"\s*(Symmetries|DATASET\s*[0-9]{1,4})\s*: space group \S* \S* \S* \(\#\S*\);\s*Bravais\s*\S*\s*\("
                  r"[a-zA-Z- .]*\)\s*$",
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
                    ] + build_abinit_vars_submatcher() + [
                    SM(r"={80}",
                       coverageIgnore=True),
                    SM(r"\s*chkinp: Checking input parameters for consistency(\.|,\s*jdtset=\s*[0-9]+\.)",
                       coverageIgnore=True, repeats=True)
                    ]
       )


SCFResultsMatcher = \
    SM(name='SCFResults',
       startReStr=r"\s*----iterations are completed or convergence reached----\s*$",
       required=False,
       subMatchers=[SM(r"\s*Mean square residual over all n,k,spin=\s*[-+0-9.eEdD]+\s*;\s*max=\s*[-+0-9.eEdD]+\s*$",
                       coverageIgnore=True),
                    SM(startReStr=r"\s*cartesian forces \(hartree/bohr\) at end:\s*$",
                       subMatchers=[SM(r"\s*[0-9]+(?P<x_abinit_atom_force_final>(\s*[-+0-9.]+){3})\s*$",
                                       repeats=True),
                                    SM(r"\s*frms,max,avg=(\s*[-+0-9.eEdD]+){5}\s*h/b\s*$")
                                    ]
                       ),
                    SM(startReStr=r"\s*cartesian forces \(eV/Angstrom\) at end:\s*$",
                       subMatchers=[SM(r"\s*[0-9]+(\s*[-+0-9.]+){3}\s*$",
                                       repeats=True),
                                    SM(r"\s*frms,max,avg=(\s*[-+0-9.eEdD]+){5}\s*e/A\s*$")
                                    ]
                       ),
                    SM(startReStr=r"\s*(Total charge density|Spin up density|Spin down density|"
                                  r"Magnetization \(spin up - spin down\)|"
                                  r"Relative magnetization \(=zeta, between -1 and 1\))\s*(\[el/Bohr\^3\])?\s*$",
                       coverageIgnore=True,
                       repeats=True,
                       subMatchers=[SM(r",(Next)?\s*(m|M)(axi|ini)mum=\s*[-+0-9.eEdD]+\s*at reduced coord."
                                       r"(\s*[0-9.]+){3}\s*$",
                                       repeats=True),
                                    SM(r",\s*Integrated=\s*[-+0-9.eEdD]+\s*$"),
                                    ]
                       ),
                    SM(r"-{80}",
                       coverageIgnore=True),
                    SM(r"\s*Components of total free energy \(in Hartree\) :\s*$",
                       coverageIgnore=True),
                    SM(r"\s*Kinetic energy\s*=\s*(?P<x_abinit_energy_kinetic>[-+0-9.eEdD]+)\s*$"),
                    SM(r"\s*Hartree energy\s*=\s*(?P<x_abinit_energy_hartree>[-+0-9.eEdD]+)\s*$"),
                    SM(r"\s*XC energy\s*=\s*(?P<x_abinit_energy_xc>[-+0-9.eEdD]+)\s*$"),
                    SM(r"\s*Ewald energy\s*=\s*(?P<x_abinit_energy_ewald>[-+0-9.eEdD]+)\s*$"),
                    SM(r"\s*PspCore energy\s*=\s*(?P<x_abinit_energy_psp_core>[-+0-9.eEdD]+)\s*$"),
                    SM(r"\s*Loc. psp. energy\s*=\s*(?P<x_abinit_energy_psp_local>[-+0-9.eEdD]+)\s*$"),
                    SM(r"\s*NL   psp  energy\s*=\s*(?P<x_abinit_energy_psp_nonlocal>[-+0-9.eEdD]+)\s*$"),
                    SM(r"\s*>{5}\s*Internal E=\s*(?P<x_abinit_energy_internal>[-+0-9.eEdD]+)\s*$"),
                    SM(r"\s*-kT\*entropy\s*=\s*(?P<x_abinit_energy_ktentropy>[-+0-9.eEdD]+)\s*$"),
                    SM(r"\s*>{9}\s*Etotal=\s*(?P<energy_total__hartree>[-+0-9.eEdD]+)\s*$"),
                    SM(r"\s*Other information on the energy :\s*$",
                       coverageIgnore=True),
                    SM(r"\s*Total energy\(eV\)=\s*[-+0-9.eEdD]+\s*;\s*Band energy \(Ha\)=\s*"
                       r"(?P<x_abinit_energy_band>[-+0-9.eEdD]+)\s*$"),
                    SM(r"-{80}",
                       coverageIgnore=True)
                    ]
       )

SCFOutput = \
    SM(name='SCFOutput',
       startReStr=r"-{3}OUTPUT-{71}\s*$",
       required=False,
       subMatchers=[SM(startReStr=r"\s*Cartesian forces \(fcart\) \[Ha/bohr\]; max,rms=(\s*[-+0-9.eEdD]+){2}\s*"
                                  r"\(free atoms\)\s*$",
                       subMatchers=[SM(r"\s*(?P<x_abinit_atom_force>(\s*[-+0-9.eEdD]+){3})\s*$",
                                       repeats=True)]
                       ),
                    SM(startReStr=r"\s*Reduced forces \(fred\)\s*",
                       subMatchers=[SM(r"\s*(\s*[-+0-9.eEdD]+){3}\s*$",
                                       repeats=True)]
                       )
                    ]
       )


SCFCycleMatcher = \
    SM(name='SCFCycle',
       startReStr=r"\s*iter\s*Etot\(hartree\)\s*deltaE\(h\)(\s*\w+)*\s*$",
       repeats=True,
       sections=['section_single_configuration_calculation'],
       subMatchers=[SM(r"\s*ETOT\s*[0-9]+\s*(?P<energy_total_scf_iteration__hartree>[-+0-9.eEdD]+)\s*"
                       r"(?P<energy_change_scf_iteration__hartree>[-+0-9.eEdD]+)(\s*[-+0-9.eEdD]*)*",
                       sections=["section_scf_iteration"],
                       repeats=True),
                    SM(r"\s*At SCF step\s*(?P<number_of_scf_iterations>[0-9]+)\s*(, etot is converged :|, forces are "
                       r"converged : |vres2\s*=\s*[-+0-9.eEdD]+\s*<\s*tolvrs=\s*[-+0-9.eEdD]+\s*=>converged.)\s*$"),
                    SM(r"\s*for the second time, (max diff in force|diff in etot)=\s*[-+0-9.eEdD]+\s*<\s*tol(dfe|dff)="
                       r"\s*[-+0-9.eEdD]+\s*$"),
                    SCFOutput,
                    SCFResultsMatcher
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
                    SM(name="pseudopotential",
                       startReStr=r"-\s*pspini: atom type\s*[0-9]+\s*psp file is \S*\s*$",
                       forwardMatch=True,
                       repeats=True,
                       coverageIgnore=True,
                       subMatchers=[SM(r"-\s*pspini: atom type\s*[0-9]+\s*psp file is\s*\S*\s*$"),
                                    SM(r"-\s*pspatm: opening atomic psp file\s*\S*",
                                       coverageIgnore=True)
                                    ]
                       ),
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
                    SCFCycleMatcher
                    ]
       )


outputVarsMatcher = \
    SM(name='OutputVars',
       startReStr=r"\s*-outvars: echo values of variables after computation  --------",
       endReStr=r"={80}",
       coverageIgnore=True,
       required=True,
       subMatchers=build_abinit_vars_submatcher(is_output=True)
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
                                    SM(r"== END DATASET\(S\) ={62}", coverageIgnore=True),
                                    SM(r"={80}", coverageIgnore=True, weak=True),
                                    outputVarsMatcher,
                                    timerMatcher,
                                    footerMatcher
                                    ]
                       )
                    ]
       )


if __name__ == "__main__":
    superContext = ABINITContext()
    mainFunction(mainFileDescription=mainFileDescription,
                 metaInfoEnv=metaInfoEnv,
                 parserInfo=parserInfo,
                 cachingLevelForMetaName={'x_abinit_section_var': CachingLevel.Cache
                                          },
                 superContext=superContext)
