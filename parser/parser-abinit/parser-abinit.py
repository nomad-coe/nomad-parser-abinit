from builtins import object
import setup_paths
from nomadcore.caching_backend import CachingLevel
from nomadcore.simple_parser import SimpleMatcher as SM
from nomadcore.simple_parser import mainFunction
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
import os, sys, json
import logging
import time

logger = logging.getLogger("nomad.ABINITParser")


class ABINITContext(object):
    """context for the sample parser"""

    def __init__(self):
        self.parser = None

    def initialize_values(self):
        """allows to reset values if the same superContext is used to parse different files"""
        self._ndatasets = 1
        self._current_dataset = 1
        self._commonInputVars = {}
        self._datasetInputVars = {}

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

#        print section["x_abinit_input_ecut"], section["x_abinit_input_nband"]

    def onClose_x_abinit_section_input(self, backend, gIndex, section):
        pass
#        backend.addValue("x_abinit_input_ecut", backend.convertScalarStringValue("x_abinit_input_ecut", "1.0"))
#        backend.addValue("x_abinit_input_nband", backend.convertScalarStringValue("x_abinit_input_nband", self._commonInputVars["nband"]))

#        print
#        print "common:", self._commonInputVars
#        print
#        print "dataset:", self._datasetInputVars

    def onClose_x_abinit_section_var(self, backend, gIndex, section):
        """Trigger called when x_abinit_section_var is closed.
        """
        if section["x_abinit_dataset"] is None:
            dataset = None
        else:
            dataset = section["x_abinit_dataset"][0]
        if dataset is None:
            if len(section["x_abinit_input_var"]) == 1:
                self._commonInputVars[section["x_abinit_varname"][0]] = section["x_abinit_input_var"][0]
            else:
                self._commonInputVars[section["x_abinit_varname"][0]] = section["x_abinit_input_var"]
        else:
            if dataset not in self._datasetInputVars.keys():
                self._datasetInputVars[dataset] = {}
            if len(section["x_abinit_input_var"]) == 1:
                self._datasetInputVars[dataset][section["x_abinit_varname"][0]] = section["x_abinit_input_var"][0]
            else:
                self._datasetInputVars[dataset][section["x_abinit_varname"][0]] = section["x_abinit_input_var"]

def build_InputVarSubMatcher(name, pattern):
    matcher = SM(startReStr=r"-?P?\s+%s(\s+|[0-9]{1,4})" % (name),
                 forwardMatch=True,
                 sections=['x_abinit_section_var'],
                 repeats=True,
                 subMatchers=[
                              SM(r"-?P?\s+%s(?P<x_abinit_dataset>[0-9]{1,4})" % (name), forwardMatch=True),
                              SM(r"-?P?\s+(?P<x_abinit_varname>%s)(\s+|[0-9]{1,4})\s+(?P<x_abinit_input_var>(%s\s*)+)\n" % (name, pattern)),
                              SM(r"\s+(?P<x_abinit_input_var>(%s\s*)+)\n" % (pattern), repeats=True),
                              SM(r"-?P?\s+[a-zA-Z_0-9]+(\s+\S*)+", forwardMatch=True)
                              ]
                 )

    return matcher


# description of the input
headerMatcher = SM(name='header',
                   startReStr="",
                   required=True,
                   subMatchers=[
                                SM(r"\.Version (?P<program_version>[0-9a-zA-Z_.]*) of ABINIT\s*"),
                                SM(r"\.\((?P<x_abinit_parallel_compilation>[a-zA-Z]*)\s*version, prepared for a (?P<program_compilation_host>\S*)\s*computer\)"),
                                SM(startReStr="\.Copyright", endReStr=r"www\.abinit\.org \.",),
                                SM(r"\.Starting date : (?P<x_abinit_start_date>[0-9a-zA-Z ]*)\."),
                                SM(r"^- \( at\s*(?P<x_abinit_start_time>[0-9a-z]*)\s*\)"),
                                SM(r"^- input  file\s*->\s*(?P<x_abinit_input_file>\S*)"),
                                SM(r"^- output file\s*->\s*(?P<x_abinit_output_file>\S*)"),
                                SM(r"^- root for input  files\s*->\s*(?P<x_abinit_input_files_root>\S*)"),
                                SM(r"^- root for output files\s*->\s*(?P<x_abinit_output_files_root>\S*)")
                                ],
                   )

timerMatcher = SM(name='timer',
                  startReStr="- Total cpu\s*time",
                  endReStr="={80}",
                  required=True,
                  forwardMatch=True,
                  subMatchers=[
                               SM(r"- Total cpu\s*time\s*\(\S*\):\s*(?P<x_abinit_total_cpu_time>[0-9.]+)\s*\S*\s*\S*"),
                               SM(r"- Total wall clock time\s*\(\S*\):\s*(?P<x_abinit_total_wallclock_time>[0-9.]+)\s*\S*\s*\S*")
                               ]
                  )

footerMatcher = SM(name='footer',
                   startReStr="\s*Suggested references for the acknowledgment of ABINIT usage.\s*",
                   endReStr="={80}",
                   required=True
                   )

memestimationMatcher = SM(name='mem_estimation',
                          startReStr=r"\s*(Symmetries|DATASET[0-9]{1,4})\s*: space group \S* \S* \S* \(\#\S*\);\s*Bravais\s*\S*\s*\([a-zA-Z- ]*\)$",
                          endReStr=r"={80}",
                          repeats=True,
                          subMatchers=[SM(r"={80}"),
                                       SM(r"\s*Values of the parameters that define the memory need of the present run"),
                                       # We ignore the values (what is print is abinit version dependent and depends
                                       # on the actual values of multiple parameters). The most important ones are
                                       # repeated later.
                                       SM(r"={80}"),
                                       SM(r"P This job should need less than\s*[0-9.]+\s*Mbytes of memory."),
                                       SM(r"\s*Rough estimation \(10\% accuracy\) of disk space for files :"),
                                       SM(r"_ WF disk file :\s*[0-9.]\s*Mbytes ; DEN or POT disk file :\s*[0-9.]\s*Mbytes."),
                                       SM(r"={80}")
                                      ]
                         )

inputMatcher = SM(name='input',
                  startReStr=r"-{80}",
                  endReStr=r"={80}",
                  forwardMatch=True,
                  sections=["x_abinit_section_input", "section_method"],
                  subMatchers=[SM(r"-{13} Echo of variables that govern the present computation -{12}"),
                               SM(r"-{80}"),
                               SM(r"-"),
                               SM(r"- outvars: echo of selected default values"),
                               SM(r"-(\s*\w+\s*=\s*[0-9]+\s*,{0,1})*"),
                               SM(r"-"),
                               SM(r"- outvars: echo of global parameters not present in the input file"),
                               SM(r"-(\s*\w+\s*=\s*[0-9]+\s*,{0,1})*"),
                               SM(r" -outvars: echo values of preprocessed input variables --------"),
                               build_InputVarSubMatcher("acell", "[-+0-9.eEdD]+\s+[-+0-9.eEdD]+\s+[-+0-9.eEdD]+\s+\w+"),
                               build_InputVarSubMatcher("amu", "[-+0-9.eEdD]+"),
                               build_InputVarSubMatcher("bs_loband", "[0-9]+"),
                               build_InputVarSubMatcher("diemac", "[-+0-9.eEdD]+"),
                               build_InputVarSubMatcher("ecut", "[-+0-9.eEdD]+\s+\w+"),
                               build_InputVarSubMatcher("fftalg", "[0-9]+"),
                               build_InputVarSubMatcher("iscf", "[0-9]+"),
                               build_InputVarSubMatcher("jdtset", "[0-9]+"),
                               build_InputVarSubMatcher("kpt", "[-+0-9.eEdD]+"),
                               build_InputVarSubMatcher("kptrlatt", "[-+0-9]+"),
                               build_InputVarSubMatcher("kptrlen", "[-+0-9.eEdD]+"),
                               build_InputVarSubMatcher("mkmem", "[0-9]+"),
                               build_InputVarSubMatcher("natom", "[0-9]+"),
                               build_InputVarSubMatcher("nband", "[0-9]+"),
                               build_InputVarSubMatcher("ndtset", "[0-9]+"),
                               build_InputVarSubMatcher("ngfft", "[0-9]+"),
                               build_InputVarSubMatcher("nkpt", "[0-9]+"),
                               build_InputVarSubMatcher("nspden", "[0-9]+"),
                               build_InputVarSubMatcher("nsppol", "[0-9]+"),
                               build_InputVarSubMatcher("nstep", "[0-9]+"),
                               build_InputVarSubMatcher("nsym", "[0-9]+"),
                               build_InputVarSubMatcher("ntypat", "[0-9]+"),
                               build_InputVarSubMatcher("occ", "[-+0-9.]+"),
                               build_InputVarSubMatcher("occopt", "[0-9]+"),
                               build_InputVarSubMatcher("prtdos", "[0-9]+"),
                               build_InputVarSubMatcher("rprim", "[-+0-9.eEdD]+"),
                               build_InputVarSubMatcher("shiftk", "[-+0-9.eEdD]+"),
                               build_InputVarSubMatcher("spgroup", "[0-9]+"),
                               build_InputVarSubMatcher("spinat", "[-+0-9.eEdD]+"),
                               build_InputVarSubMatcher("symrel", "[-+0-1]+"),
                               build_InputVarSubMatcher("tnons", "[-+0-9.]+"),
                               build_InputVarSubMatcher("toldfe", "[-+0-9.eEdD]+\s+\w+"),
                               build_InputVarSubMatcher("tolvrs", "[-+0-9.eEdD]+"),
                               build_InputVarSubMatcher("tsmear", "[-+0-9.eEdD]+\s+\w+"),
                               build_InputVarSubMatcher("typat", "[0-9]+"),
                               build_InputVarSubMatcher("wtk", "[0-9.]+"),
                               build_InputVarSubMatcher("xangst", "[-+0-9.eEdD]+"),
                               build_InputVarSubMatcher("xcart", "[-+0-9.eEdD]+"),
                               build_InputVarSubMatcher("xred", "[-+0-9.eEdD]+"),
                               build_InputVarSubMatcher("xangst", "[0-9.]+"),
                               build_InputVarSubMatcher("znucl", "[0-9.]+")
                               ]
                  )



mainFileDescription = SM(name='root',
                         startReStr="",
                         required=True,
                         subMatchers=[
                                      SM(name='NewRun',
                                         startReStr="",
                                         endReStr=r"\s*Overall time at end \(sec\) : cpu=\s*\S*\s*wall=\s*\S*",
                                         required=True,
                                         fixedStartValues={'program_name': 'ABINIT', 'program_basis_set_type': 'plane waves'},
                                         sections=['section_run'],
                                         subMatchers=[headerMatcher,
                                                      memestimationMatcher,
                                                      inputMatcher,
                                                      timerMatcher,
                                                      footerMatcher
                                                      ]
                                         )
                                      ]
                         )

# loading metadata from nomad-meta-info/meta_info/nomad_meta_info/abinit.nomadmetainfo.json

parserInfo = {
  "name": "ABINIT_parser",
  "version": "1.0"
}

metaInfoPath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "../../../../nomad-meta-info/meta_info/nomad_meta_info/abinit.nomadmetainfo.json"))
metaInfoEnv, warnings = loadJsonFile(filePath=metaInfoPath,
                                     dependencyLoader=None,
                                     extraArgsHandling=InfoKindEl.ADD_EXTRA_ARGS,
                                     uri=None)


if __name__ == "__main__":
    superContext = ABINITContext()
    mainFunction(mainFileDescription,
                 metaInfoEnv,
                 parserInfo,
                 cachingLevelForMetaName={
                                          'x_abinit_varname': CachingLevel.Cache,
                                          'x_abinit_input_var': CachingLevel.Cache,
                                          'x_abinit_section_var': CachingLevel.Cache
                                          },
                 superContext=superContext)
