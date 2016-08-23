from builtins import object
import setup_paths
from nomadcore.caching_backend import CachingLevel
from nomadcore.simple_parser import SimpleMatcher as SM
from nomadcore.simple_parser import mainFunction
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
from nomadcore import parser_backend
from nomadcore.unit_conversion import unit_conversion
import numpy as np
import re
import os, sys, json
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

    def initialize_values(self):
        """allows to reset values if the same superContext is used to parse different files"""
        self._current_dataset = 0
        # Initialize dict to store Abinit variables. Two of them are created by default:
        #  - dataset "0", which will contain the values that are common to all datasets
        #  - dataset "1", as this is the default dataset number used by Abinit when the user
        #    does not specify the dataset number
        self._abinitVars = {key: {} for key in [0, 1]}

    def startedParsing(self, filename, parser):
        """called when parsing starts"""
        self.parser = parser
        # allows to reset values if the same superContext is used to parse different files
        self.initialize_values()

    def onClose_section_run(self, backend, gIndex, section):
        """Trigger called when section_run is closed.
        """
        # Convert date and time to epoch time
        if (section["x_abinit_start_date"] is not None) and (section["x_abinit_start_time"] is not None):
            abi_time = time.strptime(str("%s %s") % (section["x_abinit_start_date"][0], section["x_abinit_start_time"][0]), "%a %d %b %Y %Hh%M")
            backend.addValue("time_run_date_start", time.mktime(abi_time))

    def onOpen_x_abinit_section_dataset(self, backend, gIndex, section):
        self._inputGIndex = backend.openSection("x_abinit_section_input")

    def onClose_x_abinit_section_dataset(self, backend, gIndex, section):
        self._current_dataset = section["x_abinit_dataset_number"][0]

        backend.closeSection("x_abinit_section_input", self._inputGIndex)

    def onOpen_x_abinit_section_input(self, backend, gIndex, section):
        self._input = section

    def onClose_x_abinit_section_input(self, backend, gIndex, section):

        dataset_vars = {}
        for varname in metaInfoEnv.infoKinds.keys():
            if "x_abinit_var_" in varname:
                dataset_vars[varname] = None

        dataset_vars.update(self._abinitVars[0])
        dataset_vars.update(self._abinitVars[self._current_dataset])

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

            metaInfo = metaInfoEnv.infoKindEl(varname)

            # Skip optional variables that do not have a value or that are not defined in the meta-info
            if varvalue is None or metaInfo is None:
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
                array = np.array(varvalue.split(), dtype=parser_backend.numpyDtypeForDtypeStr(metaInfo.dtypeStr))
                backend.addArrayValues(varname, array.reshape([nband]))

            elif len(metaInfo.shape) == 0:
                # This is a simple scalar
                backend.addValue(varname, backend.convertScalarStringValue(varname, varvalue))

            else:
                # This is an array
                array = np.array(varvalue.split(), dtype=parser_backend.numpyDtypeForDtypeStr(metaInfo.dtypeStr))
                shape = []
                for dim in metaInfo.shape:
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
        if dataset not in self._abinitVars.keys():
            self._abinitVars[dataset] = {}
        if len(section["x_abinit_varvalue"]) == 1:
            self._abinitVars[dataset]["x_abinit_var_"+section["x_abinit_varname"][0]] = section["x_abinit_varvalue"][0]
        else:
            self._abinitVars[dataset]["x_abinit_var_"+section["x_abinit_varname"][0]] = " ".join(section["x_abinit_varvalue"])


def build_AbinitVarsSubMatcher(is_output=False):
    matchers = []

    # Generate a dict of Abinit variables with the corresponding regex pattern.
    abiVars = {}
    for varname in metaInfoEnv.infoKinds.keys():
        if "x_abinit_var_" in varname:
            metaInfo = metaInfoEnv.infoKindEl(varname)
            if metaInfo.dtypeStr == "f":
                pattern = "[-+0-9.eEdD]+"
            elif metaInfo.dtypeStr == "i":
                pattern = "[-+0-9]+"
            elif metaInfo.dtypeStr == "C":
                pattern = "[\w+]"
            abiVars[re.sub("x_abinit_var_", "", varname)] = pattern

    # Some variables that Abinit writes to the output are not documented as input variables so we add them here to the
    # dictionary.
    abiVars.update({"mkmem": "[-+0-9]+"})
    if is_output:
        abiVars.update({"etotal": "[-+0-9.eEdD]+",
                        "fcart": "[-+0-9.eEdD]+",
                        "strten": "[-+0-9.eEdD]+"
                        })

    # Currently we cannot create matchers for all the Abinit input variables as this would generate more than 100 named
    # groups in regexp. Therefore we will only try to parse a subset of the input variables. This should be changed once
    # this problem is fixed
    supportedVars = ["acell", "amu", "bs_loband", "diemac", "ecut", "etotal", "fcart", "fftalg", "ionmov", "iscf",
                     "istwfk", "jdtset", "natom", "kpt", "kptopt", "kptrlatt", "kptrlen", "mkmem", "nband", "ndtset",
                     "ngfft", "nkpt", "nspden", "nsppol", "nstep", "nsym", "ntime", "ntypat", "occ", "occopt",
                     "optforces", "prtdos", "rprim", "shiftk", "spgroup", "spinat", "strten", "symafm", "symrel",
                     "tnons", "toldfe", "toldff", "tolmxf", "tolvrs", "tsmear", "typat", "wtk", "xangst", "xcart",
                     "xred", "znucl"]
    for varname in sorted(abiVars):
        if varname in supportedVars:
            matchers.append(SM(startReStr=r"[-P]?\s+%s[0-9]{0,4}\s+(%s\s*)+\s*(Hartree|Bohr)?" % (varname, abiVars[varname]),
                               forwardMatch=True,
                               sections=['x_abinit_section_var'],
                               repeats=True,
                               subMatchers=[SM(r"[-P]?\s+(?P<x_abinit_varname>%s)((?P<x_abinit_vardtset>[0-9]{1,4})\s+|\s+)"
                                               r"(?P<x_abinit_varvalue>(%s\s*)+)\s*(Hartree|Bohr)?\n" % (varname, abiVars[varname])),
                                            SM(r"\s{20,}(?P<x_abinit_varvalue>(%s\s*)+)\n" % (abiVars[varname]), repeats=True)
                                           ]
                               )
                            )
    matchers.append(SM(r""))
    return matchers


# description of the input
headerMatcher = SM(name='Header',
                   startReStr="",
                   required=True,
                   forwardMatch=True,
                   subMatchers=[SM(r"\.Version (?P<program_version>[0-9a-zA-Z_.]*) of ABINIT\s*"),
                                SM(r"\.\((?P<x_abinit_parallel_compilation>[a-zA-Z]*)\s*version, prepared for a (?P<program_compilation_host>\S*)\s*computer\)"),
                                SM(r""),
                                SM(startReStr="\.Copyright \(C\) 1998-[0-9]{4} ABINIT group .",
                                   subMatchers=[SM(r"\s*ABINIT comes with ABSOLUTELY NO WARRANTY."),
                                                SM(r"\s*It is free software, and you are welcome to redistribute it"),
                                                SM(r"\s*under certain conditions \(GNU General Public License,"),
                                                SM(r"\s*see ~abinit/COPYING or http://www.gnu.org/copyleft/gpl.txt\)."),
                                                SM(r""),
                                                SM(r"\s*ABINIT is a project of the Universite Catholique de Louvain,"),
                                                SM(r"\s*Corning Inc. and other collaborators, see ~abinit/doc/developers/contributors.txt ."),
                                                SM(r"\s*Please read ~abinit/doc/users/acknowledgments.html for suggested"),
                                                SM(r"\s*acknowledgments of the ABINIT effort."),
                                                SM(r"\s*For more information, see http://www.abinit.org ."),
                                                SM(r"")
                                                ]
                                   ),
                                SM(r"\.Starting date : (?P<x_abinit_start_date>[0-9a-zA-Z ]*)\."),
                                SM(r"^- \( at\s*(?P<x_abinit_start_time>[0-9a-z]*)\s*\)"),
                                SM(r""),
                                SM(r"^- input  file\s*->\s*(?P<x_abinit_input_file>\S*)"),
                                SM(r"^- output file\s*->\s*(?P<x_abinit_output_file>\S*)"),
                                SM(r"^- root for input  files\s*->\s*(?P<x_abinit_input_files_root>\S*)"),
                                SM(r"^- root for output files\s*->\s*(?P<x_abinit_output_files_root>\S*)"),
                                SM(r""),
                                SM(r"")
                                ],
                   )

timerMatcher = SM(name='Timer',
                  startReStr="- Total cpu\s*time",
                  endReStr="={80}",
                  required=True,
                  forwardMatch=True,
                  subMatchers=[SM(r"- Total cpu\s*time\s*\(\S*\):\s*(?P<x_abinit_total_cpu_time>[0-9.]+)\s*\S*\s*\S*"),
                               SM(r"- Total wall clock time\s*\(\S*\):\s*(?P<x_abinit_total_wallclock_time>[0-9.]+)\s*\S*\s*\S*")
                               ]
                  )

memestimationMatcher = SM(name='MemEstimation',
                          startReStr=r"\s*(Symmetries|DATASET[0-9]{1,4})\s*: space group \S* \S* \S* \(\#\S*\);\s*Bravais\s*\S*\s*\([a-zA-Z- ]*\)$",
                          endReStr=r"={80}",
                          repeats=True,
                          subMatchers=[SM(r"={80}"),
                                       SM(r"\s*Values of the parameters that define the memory need of the present run"),
                                       # We ignore the values (what is printed is abinit version dependent and depends
                                       # on the actual values of multiple parameters). The most important ones are
                                       # repeated later.
                                       SM(r"={80}"),
                                       SM(r"P This job should need less than\s*[0-9.]+\s*Mbytes of memory."),
                                       SM(r"\s*Rough estimation \(10\% accuracy\) of disk space for files :"),
                                       SM(r"_ WF disk file :\s*[0-9.]\s*Mbytes ; DEN or POT disk file :\s*[0-9.]\s*Mbytes."),
                                       SM(r"={80}")
                                      ]
                         )


inputVarsMatcher = SM(name='InputVars',
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
                                   ] + build_AbinitVarsSubMatcher()
                      )

SCFCycleMatcher = SM(name='SCFCycle',
                     startReStr=r"\s*iter\s*Etot\(hartree\)\s*deltaE\(h\)(\s*\w+)*",
                     repeats=True,
                     sections=['section_single_configuration_calculation'],
                     subMatchers=[SM(r"\s*ETOT\s*[0-9]+\s*(?P<energy_total_scf_iteration>[-+0-9.eEdD]+)\s*(?P<energy_change_scf_iteration>[-+0-9.eEdD]+)(\s*[-+0-9.eEdD]*)*",
                                     sections=["section_scf_iteration"], repeats=True),
                                  SM(r"\s*At SCF step\s*[0-9]+"),
                                  SM(r"\s*>{9}\s*Etotal=\s*(?P<energy_total__hartree>[-+0-9.eEdD]+)")
                                  ]
                     )

datasetMatcher = SM(name='Dataset',
                    startReStr=r"={2}\s*DATASET\s*[0-9]+\s*={66}",
                    forwardMatch=True,
                    repeats=True,
                    sections=['section_method', 'section_system', 'x_abinit_section_dataset'],
                    subMatchers=[SM("={2}\s*DATASET\s*(?P<x_abinit_dataset_number>[0-9]+)\s*={66}"),
                                 SCFCycleMatcher
                                 ]
                    )


outputVarsMatcher = SM(name='OutputVars',
                       startReStr=r"\s*-outvars: echo values of variables after computation  --------",
                       endReStr=r"={80}",
                       required=True,
                       subMatchers=build_AbinitVarsSubMatcher(is_output=True)
                       )

footerMatcher = SM(name='Footer',
                   startReStr="\s*Suggested references for the acknowledgment of ABINIT usage.\s*",
                   required=True,
                   subMatchers=[SM(r""),
                                SM(r"\s*The users of ABINIT have little formal obligations with respect to the ABINIT group"),
                                SM(r"\s*\(those specified in the GNU General Public License, http://www.gnu.org/copyleft/gpl.txt\)."),
                                SM(r"\s*However, it is common practice in the scientific literature,"),
                                SM(r"\s*to acknowledge the efforts of people that have made the research possible."),
                                SM(r"\s*In this spirit, please find below suggested citations of work written by ABINIT developers,"),
                                SM(r"\s*corresponding to implementations inside of ABINIT that you have used in the present run."),
                                SM(r"\s*Note also that it will be of great value to readers of publications presenting these results,"),
                                SM(r"\s*to read papers enabling them to understand the theoretical formalism and details"),
                                SM(r"\s*of the ABINIT implementation."),
                                SM(r"\s*For information on why they are suggested, see also http://www.abinit.org/about/\?text=acknowledgments."),
                                SM(r"={80}"),
                                SM(r"", weak=True),
                                SM(r"\s*Calculation completed.")
                                ]
                   )

mainFileDescription = SM(name='root',
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
