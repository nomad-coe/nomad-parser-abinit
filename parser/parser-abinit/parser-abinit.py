from builtins import object
from nomadcore.caching_backend import CachingLevel
from nomadcore.simple_parser import SimpleMatcher as SM
from nomadcore.simple_parser import mainFunction
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
from nomadcore import parser_backend
from nomadcore.unit_conversion import unit_conversion
from ase.data import chemical_symbols
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
        self.inputGIndex = None

    def initialize_values(self):
        """allows to reset values if the same superContext is used to parse different files"""
        self.current_dataset = 0
        # Initialize dict to store Abinit variables. Two of them are created by default:
        #  - dataset "0", which will contain the values that are common to all datasets
        #  - dataset "1", as this is the default dataset number used by Abinit when the user
        #    does not specify the dataset number
        self.abinitVars = {key: {} for key in [0, 1]}
        self.inputGIndex = None
        self.input = None

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
            abi_time = time.strptime(str("%s %s") % (section["x_abinit_start_date"][0],
                                                     section["x_abinit_start_time"][0]), "%a %d %b %Y %Hh%M")
            backend.addValue("time_run_date_start", time.mktime(abi_time))

    def onClose_section_method(self, backend, gIndex, section):
        """Trigger called when section_method is closed.
        """
        backend.addValue("number_of_spin_channels", self.input["x_abinit_var_nsppol"])
        backend.addValue("scf_max_iteration", self.input["x_abinit_var_nstep"])
        if self.input["x_abinit_var_toldfe"] is not None:
            backend.addValue("scf_threshold_energy_change",
                             unit_conversion.convert_unit(self.input["x_abinit_var_toldfe"][0], 'hartree'))
        backend.addValue("self_interaction_correction_method", "")
        if self.input["x_abinit_var_occopt"][0] == 3:
            smear_kind = "fermi"
        elif self.input["x_abinit_var_occopt"][0] == 4 or self.input["x_abinit_var_occopt"] == 5:
            smear_kind = "marzari-vanderbilt"
        elif self.input["x_abinit_var_occopt"][0] == 6:
            smear_kind = "methfessel-paxton"
        elif self.input["x_abinit_var_occopt"][0] == 7:
            smear_kind = "gaussian"
        elif self.input["x_abinit_var_occopt"][0] == 8:
            logger.error("Illegal value for Abinit input variable occopt")
            smear_kind = ""
        else:
            smear_kind = ""
        backend.addValue("smearing_kind", smear_kind)
        if self.input["x_abinit_var_tsmear"] is not None:
            backend.addValue("smearing_width",
                             unit_conversion.convert_unit(self.input["x_abinit_var_tsmear"][0], 'hartree'))

    def onClose_section_system(self, backend, gIndex, section):
        """Trigger called when section_system is closed.
        """
        species_count = {}
        for z in self.input["x_abinit_var_znucl"][0]:
            species_count[chemical_symbols[int(z)]] = 0
        atom_types = []
        for z in self.input["x_abinit_var_znucl"][0]:
            symbol = chemical_symbols[int(z)]
            species_count[symbol] += 1
            atom_types.append(symbol+str(species_count[symbol]))
        atom_labels = backend.arrayForMetaInfo("atom_labels", self.input["x_abinit_var_natom"])
        for atom_index in range(self.input["x_abinit_var_natom"][0]):
            atom_labels[atom_index] = atom_types[self.input["x_abinit_var_typat"][0][atom_index] - 1]
        backend.addArrayValues("atom_labels", atom_labels)

        if self.input["x_abinit_var_xcart"] is None:
            if self.input["x_abinit_var_natom"][0] == 1:
                backend.addArrayValues("atom_positions", np.array([[0, 0, 0]]))
            else:
                logger.error("Positions of atoms is not available")
        else:
            backend.addArrayValues("atom_positions", self.input["x_abinit_var_xcart"][0])

        backend.addArrayValues("configuration_periodic_dimensions", np.array([True, True, True]))

        backend.addValue("number_of_atoms", self.input["x_abinit_var_natom"])

        backend.addValue("spacegroup_3D_number", self.input["x_abinit_var_spgroup"])

    def onOpen_x_abinit_section_dataset(self, backend, gIndex, section):
        """Trigger called when x_abinit_section_dataset is opened.
        """
        self.inputGIndex = backend.openSection("x_abinit_section_input")

    def onClose_x_abinit_section_dataset(self, backend, gIndex, section):
        """Trigger called when x_abinit_section_dataset is closed.
        """
        self.current_dataset = section["x_abinit_dataset_number"][0]

        backend.closeSection("x_abinit_section_input", self.inputGIndex)

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
    # this problem is fixed
    supported_vars = \
        ["acell", "amu", "bs_loband", "diemac", "ecut", "etotal", "fcart", "fftalg", "ionmov", "iscf", "istwfk", "ixc",
         "jdtset", "natom", "kpt", "kptopt", "kptrlatt", "kptrlen", "mkmem", "nband", "ndtset", "ngfft", "nkpt",
         "nspden", "nsppol", "nstep", "nsym", "ntime", "ntypat", "occ", "occopt", "optforces", "prtdos", "rprim",
         "shiftk", "spgroup", "spinat", "strten", "symafm", "symrel", "tnons", "toldfe", "toldff", "tolmxf", "tolvrs",
         "tsmear", "typat", "wtk", "xangst", "xcart", "xred", "znucl"]
    for varname in sorted(abi_vars):
        if varname in supported_vars:
            matchers.append(SM(startReStr=r"[-P]?\s+%s[0-9]{0,4}\s+(%s\s*)+\s*(Hartree|Bohr)?"
                                          % (varname, abi_vars[varname]),
                               forwardMatch=True,
                               sections=['x_abinit_section_var'],
                               repeats=True,
                               subMatchers=[SM(r"[-P]?\s+(?P<x_abinit_varname>%s)((?P<x_abinit_vardtset>[0-9]{1,4})\s+|"
                                               r"\s+)(?P<x_abinit_varvalue>(%s\s*)+)\s*(Hartree|Bohr)?\n"
                                               % (varname, abi_vars[varname])),
                                            SM(r"\s{20,}(?P<x_abinit_varvalue>(%s\s*)+)\n" % (abi_vars[varname]),
                                               repeats=True)
                                            ]
                               )
                            )
    matchers.append(SM(r""))
    return matchers


# description of the input
headerMatcher = \
    SM(name='Header',
       startReStr="",
       required=True,
       forwardMatch=True,
       subMatchers=[SM(r"\.Version (?P<program_version>[0-9a-zA-Z_.]*) of ABINIT\s*"),
                    SM(r"\.\((?P<x_abinit_parallel_compilation>[a-zA-Z]*)\s*version, prepared for a "
                       r"(?P<program_compilation_host>\S*)\s*computer\)"),
                    SM(r""),
                    SM(startReStr="\.Copyright \(C\) 1998-[0-9]{4} ABINIT group .",
                       subMatchers=[SM(r"\s*ABINIT comes with ABSOLUTELY NO WARRANTY."),
                                    SM(r"\s*It is free software, and you are welcome to redistribute it"),
                                    SM(r"\s*under certain conditions \(GNU General Public License,"),
                                    SM(r"\s*see ~abinit/COPYING or http://www.gnu.org/copyleft/gpl.txt\)."),
                                    SM(r""),
                                    SM(r"\s*ABINIT is a project of the Universite Catholique de Louvain,"),
                                    SM(r"\s*Corning Inc. and other collaborators, see "
                                       r"~abinit/doc/developers/contributors.txt ."),
                                    SM(r"\s*Please read ~abinit/doc/users/acknowledgments.html for suggested"),
                                    SM(r"\s*acknowledgments of the ABINIT effort."),
                                    SM(r"\s*For more information, see http://www.abinit.org .")
                                    ]
                       ),
                    SM(r"\.Starting date : (?P<x_abinit_start_date>[0-9a-zA-Z ]*)\."),
                    SM(r"^- \( at\s*(?P<x_abinit_start_time>[0-9a-z]*)\s*\)"),
                    SM(r"^- input  file\s*->\s*(?P<x_abinit_input_file>\S*)"),
                    SM(r"^- output file\s*->\s*(?P<x_abinit_output_file>\S*)"),
                    SM(r"^- root for input  files\s*->\s*(?P<x_abinit_input_files_root>\S*)"),
                    SM(r"^- root for output files\s*->\s*(?P<x_abinit_output_files_root>\S*)")
                    ],
       )

timerMatcher = \
    SM(name='Timer',
       startReStr="- Total cpu\s*time",
       endReStr="={80}",
       required=True,
       forwardMatch=True,
       subMatchers=[SM(r"- Total cpu\s*time\s*\(\S*\):\s*(?P<x_abinit_total_cpu_time>[0-9.]+)\s*\S*\s*\S*"),
                    SM(r"- Total wall clock time\s*\(\S*\):\s*(?P<x_abinit_total_wallclock_time>[0-9.]+)\s*\S*\s*\S*")
                    ]
       )

memestimationMatcher = \
    SM(name='MemEstimation',
       startReStr=r"\s*(Symmetries|DATASET\s*[0-9]{1,4})\s*: space group \S* \S* \S* \(\#\S*\);\s*Bravais\s*\S*\s*\("
                  r"[a-zA-Z- .]*\)$",
       endReStr=r"={80}",
       repeats=True,
       subMatchers=[SM(r"={80}",
                       coverageIgnore=True),
                    SM(r"\s*Values of the parameters that define the memory need (of the present run|for DATASET\s*"
                       r"[0-9]+\.)"),
                    # We ignore the values (what is printed is abinit version dependent and depends
                    # on the actual values of multiple parameters). The most important ones are
                    # repeated later.
                    SM(r"={80}"),
                    SM(r"P This job should need less than\s*[0-9.]+\s*Mbytes of memory."),
                    SM(r"\s*Rough estimation \(10\% accuracy\) of disk space for files :"),
                    SM(r"_ WF disk file :\s*[0-9.]+\s*Mbytes ; DEN or POT disk file :\s*[0-9.]+\s*Mbytes."),
                    SM(r"={80}")
                    ]
       )


inputVarsMatcher = \
    SM(name='InputVars',
       startReStr=r"-{80}",
       endReStr=r"={80}",
       required=True,
       subMatchers=[SM(r"-{13} Echo of variables that govern the present computation -{12}"),
                    SM(r"-{80}"),
                    SM(r"-"),
                    SM(r"- outvars: echo of selected default values"),
                    SM(r"-(\s*\w+\s*=\s*[0-9]+\s*,{0,1})*"),
                    SM(r"-"),
                    SM(r"- outvars: echo of global parameters not present in the input file"),
                    SM(r"-(\s*\w+\s*=\s*[0-9]+\s*,{0,1})*"),
                    SM(r"-"),
                    SM(r" -outvars: echo values of preprocessed input variables --------"),
                    ] + build_abinit_vars_submatcher() + [
                    SM(r"={80}",
                       coverageIgnore=True),
                    SM(r"\s*chkinp: Checking input parameters for consistency(\.|,\s*jdtset=\s*[0-9]+\.)",
                       repeats=True)
                    ]
       )

SCFCycleMatcher = \
    SM(name='SCFCycle',
       startReStr=r"\s*iter\s*Etot\(hartree\)\s*deltaE\(h\)(\s*\w+)*",
       repeats=True,
       sections=['section_single_configuration_calculation'],
       subMatchers=[SM(r"\s*ETOT\s*[0-9]+\s*(?P<energy_total_scf_iteration>[-+0-9.eEdD]+)\s*"
                       r"(?P<energy_change_scf_iteration>[-+0-9.eEdD]+)(\s*[-+0-9.eEdD]*)*",
                       sections=["section_scf_iteration"],
                       repeats=True),
                    SM(r"\s*At SCF step\s*(?P<number_of_scf_iterations>[0-9]+)\s*(, etot is converged :|, forces are "
                       r"converged : |vres2\s*=\s*[-+0-9.eEdD]+\s*<\s*tolvrs=\s*[-+0-9.eEdD]+\s*=>converged.)"),
                    SM(r"\s*for the second time, (max diff in force|diff in etot)=\s*[-+0-9.eEdD]+\s*<\s*tol(dfe|dff)="
                       r"\s*[-+0-9.eEdD]+"),
                    SM(r"\s*>{9}\s*Etotal=\s*(?P<energy_total__hartree>[-+0-9.eEdD]+)")
                    ]
       )

datasetMatcher = \
    SM(name='Dataset',
       startReStr=r"={2}\s*DATASET\s*[0-9]+\s*={66}",
       forwardMatch=True,
       repeats=True,
       sections=['section_method', 'section_system', 'x_abinit_section_dataset'],
       subMatchers=[SM("={2}\s*DATASET\s*(?P<x_abinit_dataset_number>[0-9]+)\s*={66}"),
                    SM("-\s*nproc\s*=\s*[0-9]+"),
                    SCFCycleMatcher
                    ]
       )


outputVarsMatcher = \
    SM(name='OutputVars',
       startReStr=r"\s*-outvars: echo values of variables after computation  --------",
       endReStr=r"={80}",
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
                    SM(r"-?\s*And optionally\s*:",
                       coverageIgnore=True),
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
       startReStr="",
       required=True,
       subMatchers=[SM(name='NewRun',
                       startReStr="",
                       endReStr=r"\s*Overall time at end \(sec\) : cpu=\s*\S*\s*wall=\s*\S*",
                       required=True,
                       fixedStartValues={'program_name': 'ABINIT', 'program_basis_set_type': 'plane waves'},
                       sections=['section_run'],
                       subMatchers=[headerMatcher,
                                    memestimationMatcher,
                                    inputVarsMatcher,
                                    datasetMatcher,
                                    outputVarsMatcher,
                                    timerMatcher,
                                    footerMatcher
                                    ]
                       )
                    ]
       )


if __name__ == "__main__":
    superContext = ABINITContext()
    mainFunction(mainFileDescription,
                 metaInfoEnv,
                 parserInfo,
                 cachingLevelForMetaName={'x_abinit_section_var': CachingLevel.Cache
                                          },
                 superContext=superContext)
