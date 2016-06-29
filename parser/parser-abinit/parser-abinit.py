from builtins import object
import setup_paths
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
        pass

    def startedParsing(self, filename, parser):
        """called when parsing starts"""
        self.parser = parser
        # allows to reset values if the same superContext is used to parse different files
        self.initialize_values()

    def onClose_section_run(self, backend, gIndex, section):
        """Trigger called when section_run is closed.
        """
        # Convert date and time to epoch time
        abi_time = time.strptime(str("%s %s")%(section["x_abinit_start_date"][0], section["x_abinit_start_time"][0]), "%a %d %b %Y %Hh%M")
        backend.addValue("time_run_date_start", time.mktime(abi_time))


# description of the input


mainFileDescription = SM(name='root',
                         weak=True,
                         startReStr="",
                         subMatchers=[
                                      SM(name='NewRun',
                                         startReStr="",
                                         endReStr=r"\s*Overall time at end",
                                         repeats=False,
                                         required=True,
                                         forwardMatch=True,
                                         fixedStartValues={'program_name': 'ABINIT', 'program_basis_set_type': 'plane waves'},
                                         sections=['section_run'],
                                         subMatchers=[SM(r"\.Version (?P<program_version>[0-9a-zA-Z_.]*) of ABINIT\s*"),
                                                      SM(r"\.\((?P<x_abinit_parallel_compilation>[a-zA-Z]*)\s*version, prepared for a (?P<program_compilation_host>\S*)\s*computer\)"),
                                                      SM(r"\.Starting date : (?P<x_abinit_start_date>[0-9a-zA-Z ]*)\."),
                                                      SM(r"^- \( at\s*(?P<x_abinit_start_time>[0-9a-z]*)\s*\)"),
                                                      SM(r"^- input  file\s*->\s*(?P<x_abinit_input_file>\S*)"),
                                                      SM(r"^- output file\s*->\s*(?P<x_abinit_output_file>\S*)"),
                                                      SM(r"^- root for input  files\s*->\s*(?P<x_abinit_input_files_root>\S*)"),
                                                      SM(r"^- root for output files\s*->\s*(?P<x_abinit_output_files_root>\S*)"),
                                                      SM(r"\s*Symmetries : space group \S* \S* \S* \(#(?P<spacegroup_3D_number>[0-9]*)\)", sections=["section_system"])
                                                      ],
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
    mainFunction(mainFileDescription, metaInfoEnv, parserInfo, superContext=superContext)
