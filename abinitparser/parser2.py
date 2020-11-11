# Copyright 2018 Markus Scheidgen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an"AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import glob
import os
import re
import numpy as np
import pint
from nomad.units import ureg
from ase.data import chemical_symbols
from nomad.parsing.text_parser import Quantity, UnstructuredTextFileParser

from abinitparser.AbinitXC import ABINIT_NATIVE_IXC, ABINIT_LIBXC_IXC
# from abinitparser.metainfo.abinit import x_abinit_section_dataset_header

from nomad.datamodel.metainfo.public import (
    section_run, section_system, section_method, section_basis_set_cell_dependent,
    section_dos, section_k_band, section_k_band_segment, section_eigenvalues,
    section_single_configuration_calculation, section_XC_functionals,
    section_symmetry, section_scf_iteration
    )

from nomad.datamodel.metainfo.common import (
    section_method_basis_set)

import logging
logger = logging.getLogger("nomad.ABINITParser")


class AbinitOutputParser(UnstructuredTextFileParser):
    """Parser of an ABINIT file"""
    def __init__(self, mainfile, logger):
        super().__init__(mainfile, None, logger)

    def init_quantities(self):
        self._quantities = []
        self._quantities.append(Quantity('program_version', r'\.Version\s*([\d\.]+)', repeats=False),)
        self._quantities.append(Quantity('root_out', r'- root for output files ->\s*([\w]+)', repeats=False))
        #                                              - root for output files -> tspin_1o

        def string_to_ener(string):
            string = string.replace('\n\n', '\n') # remove empty lines
            val = [v.split('=') for v in string.strip().split('\n')]
            for v in val:
                v[0] = v[0].strip().lstrip('>').strip()
            return {v[0] : pint.Quantity(float(v[1]),'hartree') for v in val}

        def string_to_kpt(string):
            val = string.strip().split('kpt')
            kpoints = []
            for item in val:
                item = item.split()
                if len(item) < 3:
                    continue
                item = item[1:] if item[0].isdecimal() else item
                kpoints.append(np.reshape(item, (len(item)//3, 3)))
            return kpoints

        def string_to_symrel(string):
            val = string.strip().split('symrel')
            symrels = val[1].strip().split()
            symrel_list = [int(item) for item in symrels]
            # reshape such that first index is a matrix handle
            return np.reshape(symrel_list, (-1, 3, 3))

        # Common regex's
        int_re = r'([\-\d]+)'
        float_re = r'([\d\.E\+\-]+)\s*'
        float_re2 = r'([\d\.E\+\-]+)'
        array_re = r'([\d\.E\+\-\s]+)'

        # INTEGER Variables
        int_list = [
            'ndtset', 'fftalg', 'getden', 'getwfk', 'iscf',
            'mkmem', 'natom', 'nband', 'nbdbuf', 'nkpt',
            'nstep', 'nsym', 'ntypat', 'prtdos', 'spgroup']

        for key in int_list:
            # added '[\d]' to account for possible index: nkpt1, nkpt2, ...
            regex = r'%s[\d]*\s+%s' %(key, int_re)
            self._quantities.append(Quantity(key, regex, repeats=True))

        # FLOAT Variables
        float_list = [
            'amu', 'diemac', 'dosdeltae', 'ecut', 'toldfe', 'tolvrs', 'tolwfr',
            'tsmear','kptrlen']
        for key in float_list:
            regex = r'%s[\d]*\s+%s' %(key, float_re)
            self._quantities.append(Quantity(key, regex, repeats=True))

        # ARRAY Variables
        array_list =[
            'acell', 'rprim', 'tnons', 'symrel', 'istwfk', 'typat', 'xangst',
            'xcart', 'xred', 'znucl']
        for key in array_list:
            #repat = r'%s\s*%s' %(key, array_re)
            repat = r'%s[\d]*\s+%s' %(key, array_re)
            self._quantities.append(Quantity(key, repat, repeats=False))

        # SPECIAL CASES

        # input file header
        self._quantities.append(Quantity('kpt', r'-outvars: echo values of preprocessed input variables --------[\s\S]*?(kpt[kpt\d\.E\+\-\s]+)',str_operation=string_to_kpt, repeats=False))
        self._quantities.append(Quantity('symrel', r'-outvars: echo values of preprocessed input variables --------[\s\S]*?(symrel[symrel\d\-\s]+)',str_operation=string_to_symrel, repeats=False))

        # dataset header
        self._quantities.append(Quantity('nsppol', r'\s*nsppol\s*=\s*(\d)', repeats=False))
        self._quantities.append(Quantity('occopt', r'\s*occopt\s*=\s*(\d)', repeats=True))
        self._quantities.append(Quantity('ixc', r'[\s\S]*ixc=([\-\d]*)|ixc\s*([\-\d]*)', repeats=True))
        self._quantities.append(Quantity('iscf_fallback', r'iscf\s*=\s*([\-\d]*)', repeats=True))


        # SUBPARSING for DATASETS
        def str_to_scf_ener(string):
            # pick third column if line starts with 'ETOT'
            val = [v.split()[2] for v in string.strip().split('\n') if 'ETOT' in v]
            return val

        def str_to_force(string):
            val = string.strip().split()
            val = np.reshape(val, (-1,4))
            return val[:, 1:]   # skip first column (it's an atom index)

        def str_to_xcart(string):
            val = string.strip().split()
            val = np.reshape(val, (-1,4))
            return val[:, 1:]  # skip first column (it's an atom index)

        def str_to_stress(stress):
            pieces = stress.strip().split()
            stress = np.zeros((3,3))
            stress[0,0] = float(pieces[0])
            stress[2,1] = float(pieces[1])
            stress[1,2] = float(pieces[1])

            stress[1,1] = float(pieces[2])
            stress[2,0] = float(pieces[3])
            stress[0,2] = float(pieces[3])

            stress[2,2] = float(pieces[4])
            stress[1,0] = float(pieces[5])
            stress[0,1] = float(pieces[5])
            return pint.Quantity(stress, 'hartree/bohr**3')

        dataset_quantities = Quantity(
            'dataset',r' Exchange-correlation([\s\S]*?)(?:== DATASET  \d+|== END DATASET)',
            repeats=True,
            sub_parser=UnstructuredTextFileParser(quantities=[
                Quantity(
                    'energy_scf',
                    r'\n *iter([\s\S]*?)converged', str_operation=str_to_scf_ener,
                    repeats=False),
                Quantity(
                    'energy_components',
                    r'Components of total free energy \(in Hartree\) :([\s\S]+?Etotal=\s*[\d\.E\+\-]+)',
                    str_operation=string_to_ener, repeats=False),
                Quantity(
                    'stress',
                    r'\s* Cartesian components of stress tensor \(hartree[/]bohr.3\)'
                    r'\s*sigma\(\d \d\)=\s*([\+\-\d\.Ee]+)'
                    r'\s*sigma\(\d \d\)=\s*([\+\-\d\.Ee]+)'
                    r'\s*sigma\(\d \d\)=\s*([\+\-\d\.Ee]+)'
                    r'\s*sigma\(\d \d\)=\s*([\+\-\d\.Ee]+)'
                    r'\s*sigma\(\d \d\)=\s*([\+\-\d\.Ee]+)'
                    r'\s*sigma\(\d \d\)=\s*([\+\-\d\.Ee]+)',
                    str_operation=str_to_stress,
                    repeats=False),
                Quantity(
                    'ucvol', # TODO: archive. FIXME: issues storing a Pint quantity
                    r'\s*Unit cell volume ucvol=\s*([\+\-\d\.Ee]+)',
                    repeats=False),
                Quantity(
                    'EIG_file',
                    r'\s*prteigrs : about to open file\s*([\w]+)', repeats=False),
                Quantity(
                    'forces_SCF',
                    r'\s*cartesian forces \(hartree/bohr\) at end:\s*([\+\-\d\.\n ]+)',
                    str_operation=str_to_force, dtype=float),
                Quantity(
                    'lattice_vec',
                    r'R\(\d\)=\s*([ \d\.Ed\-]+)', repeats=True),
                Quantity(
                    'xcart_dataset',
                    r'\s*cartesian coordinates \(angstrom\) at end:\s*\n([\-\d\. \n]*)',
                    str_operation=str_to_xcart, unit='angstrom', repeats=False
                )
                    ]))
        self._quantities.append(dataset_quantities)

        # SUBPARSING the tail of the output file.
        # that is, everything below
        #      "-outvars: echo values of variables after computation", notice "AFTER".
        def string_to_nsppol(string):
            # print(string)
            return int(string.strip())

        def string_to_occ2(string):
            string = string.replace('\n\n', '\n') # remove empty lines
            #print('OCC STRING: ', string)
            val = string.strip().split()
            #print('\nVALUES', val, '\n\n')
            #print('****')
            # for x in val:
            #     print(x)
            values = [float(item) for item in val]
            #print('\nOCC VALUES:', values)
            return values

        tail_quantities = Quantity(
            'tail_report',r'\s*-outvars: echo values of variables after computation([\s\S]*?)(?:\s*Suggested references)',
            repeats=False,
            sub_parser=UnstructuredTextFileParser(quantities=[
                Quantity('t_nsppol', r'\s*nsppol[\d]?\s+([\d]+)', str_operation=string_to_nsppol, repeats=True),
                Quantity('t_occ', r'\s*occ[\d]?\s+([\d\s\-\.]+)', str_operation=string_to_occ2, dtype=float, repeats=True),
                    ]))
        #self._quantities.append(tail_quantities)


class AbinitParserInterface:
    """Class to write to NOMAD's Archive"""
    def __init__(self, mainfile, archive, logger):
        self.mainfile = mainfile
        self.archive = archive
        self.logger = logger
        self.out_parser = AbinitOutputParser(mainfile, logger)

        # map ABINIT energy names to NOMAD Metadata
        self._energy_map = {
            'Kinetic energy': 'electronic_kinetic_energy',
            'Hartree energy':'energy_correction_hartree',
            'XC energy': 'energy_XC',
            'Etotal': 'energy_total',
            # abinit-specific names:
            'PspCore energy': 'x_abinit_energy_psp_core',
            'Ewald energy': 'x_abinit_energy_ewald',
            'Loc. psp. energy': 'x_abinit_energy_psp_local',
            'NL   psp  energy': 'x_abinit_energy_psp_nonlocal'}

    def parse_method(self):
        """
        Parse and store quantities for section_method; these are the parameters
        that define theory & approximations for a single_configuration_calculation
        """
        sec_run = self.archive.section_run[-1]
        sec_method = sec_run.m_create(section_method)

        # some variables should be trim up to `ndtset`
        ndtset = self.out_parser.get('ndtset')[0]

        sec_method.electronic_structure_method = 'DFT'
        sec_method.stress_tensor_method = 'analytic'
        sec_method.number_of_spin_channels = self.out_parser.get('nsppol')
        sec_method.scf_max_iteration = int(self.out_parser.get('nstep')[0])

        sec_method.self_interaction_correction_method = "" # empty str in original parser

        # Method tolerances
        toldfe = self.out_parser.get('toldfe')
        tolvrs = self.out_parser.get('tolvrs')
        tolwfr = self.out_parser.get('tolwfr')

        if toldfe is not None:
            sec_method.scf_threshold_energy_change = pint.Quantity(toldfe, 'hartree')
        if tolvrs is not None:
            sec_method.x_abinit_tolvrs = tolvrs
        if tolwfr is not None:
            sec_method.x_abinit_tolwfr = tolwfr

        # wavefunction storage
        istwfk = self.out_parser.get('istwfk')
        if istwfk is not None:
            sec_method.x_abinit_istwfk = istwfk

        # method for self-consistent field cycles
        iscf = self.out_parser.get('iscf')
        if iscf is None:
            iscf = self.out_parser.get('iscf_fallback')
        if iscf is not None:
            iscf = np.array(iscf[:ndtset], dtype=int)
            sec_method.x_abinit_iscf = iscf

        # SMEAR KIND
        occopt = self.out_parser.get('occopt')[0]
        if occopt is not None:
            if occopt == 3:
                smear_kind = "fermi"
            elif occopt == 4 or occopt == 5:
                smear_kind = "marzari-vanderbilt"
            elif occopt == 6:
                smear_kind = "methfessel-paxton"
            elif occopt == 7:
                smear_kind = "gaussian"
            elif occopt == 8:
                logger.error("Illegal value for Abinit input variable occopt")
                smear_kind = ""
            else:
                smear_kind = ""
            sec_method.smearing_kind = smear_kind
        else:
            logger.warning(
                'Unable to parse `occopt` variable, hence unable to determine `smearing method`')

        tsmear = self.out_parser.get('tsmear')
        if tsmear is not None:
            sec_method.smearing_width = tsmear[0]

        ixc = int(self.out_parser.get('ixc')[-1])
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
                sec_xcfun = sec_method.m_create(section_XC_functionals)
                for key, value in sorted(xc_functional.items()):
                    setattr(sec_xcfun, key, value)

        sec_basisset = sec_method.m_create(section_method_basis_set)
        sec_basisset.method_basis_set_kind = "wavefunction"

    def parse_tail(self, tail):
        """ Parses an UnstructuredTextFileParser object and stores relevant data
        into NOMAD's Archive

        Args:
            tail: UnstructuredTextFileParser Object
        """
        t_nsppol = tail.get('t_nsppol')
        print('\ntail: t_nsppol: ', t_nsppol)

        t_occ = tail.get('t_occ')
        print('\ntail: t_occ:', t_occ)

    def parse_dataset(self, dataset, jdtset):
        """ Parses an UnstructuredTextFileParser object and stores relevant data
        into NOMAD's Archive

        Args:
            dataset: UnstructuredTextFileParser Object
            jdtset (int): dataset index (i.e., sscc index)
        """
        # sec_run:    a single call to a program
        # sec_system: physical properties that define a system
        # sec_scc:    values computed during a single config calc,
        #             for a given method and a given system

        sec_run = self.archive.section_run[-1]
        sec_system = sec_run.m_create(section_system)
        sscc = sec_run.m_create(section_single_configuration_calculation)

        ndtset = self.out_parser.get('ndtset')[0]

        ucvol = dataset.get('ucvol')
        if ucvol is not None:
            # print('UCVOL', ucvol, type(ucvol))
            # MARKUS: ALVIN: I can only attach unit at the end, why?
            sscc.x_abinit_unit_cell_volume = ureg.Quantity(ucvol, 'bohr**3') # works!
            # metainfo: 'dependencies/parsers/abinit/abinitparser/metainfo/abinit.py'
            # compare with 'lat_vec'

        spgroup = self.out_parser.get('spgroup')
        sec_symm = sec_system.m_create(section_symmetry)
        sec_symm.space_group_number = spgroup

        sec_system.configuration_periodic_dimensions = np.array([True, True, True])
        sec_system.number_of_atoms = int(self.out_parser.get('natom')[-1])

        lat_vec = dataset.get('lattice_vec', unit='bohr')
        # print('LAT_VEC', lat_vec)
        sec_system.lattice_vectors = lat_vec
        sec_system.simulation_cell = lat_vec

        # ENERGY COMPONENTS
        energies = dataset.get('energy_components', {})
        for key, val in energies.items():
            if key in self._energy_map:
                metainfokey = self._energy_map[key]
                if metainfokey.startswith('x_abinit'):
                    val = val.to('joule').magnitude
                setattr(sscc, self._energy_map[key], val)  #  Hartree

        # energy_total_scf_iteration
        energy_scf = dataset.get('energy_scf')
        if energy_scf is not None:
            sec_scf = sscc.m_create(section_scf_iteration)
            sec_scf.energy_total_scf_iteration = energy_scf

        # FORCES AND STRESS
        # MARKUS & ALVIN: safest unit attachment seems to be at the end
        # forces_scf = dataset.get('forces_SCF', unit='hartree / bohr') # hartree/bohr
        forces_scf = dataset.get('forces_SCF')
        if forces_scf is not None:
            # sscc.x_abinit_atom_force_final = forces_scf # fails if it's a Pint quantity
            sscc.x_abinit_atom_force_final = ureg.Quantity(forces_scf, 'hartree / bohr') # OK

        stress = dataset.get('stress')
        if stress is not None:
            sscc.stress_tensor = stress

        # ATOM POSITIONS
        xcart_ds = dataset.get('xcart_dataset')
        if xcart_ds is not None:
            sec_system.atom_positions = xcart_ds  # unit was attached during parsing
        else:
            logger.error("Atom positions within dataset are unavailable")

        # ATOM LABELS
        znucl = self.out_parser.get('znucl')
        natom = self.out_parser.get('natom')[:ndtset]
        ntypat = self.out_parser.get('ntypat')
        typat = self.out_parser.get('typat')
        # sec_system.atom_atom_number = znucl # FIXME: needs generalization

        if type(znucl) is not list:
            # enforce znucl to be a list when there's only one atom type
            znucl = [znucl]

        species_count = {}
        # collect all chemical symbols
        for z in znucl:
            species_count[chemical_symbols[int(z)]] = 0

        atom_types = []
        # pair chemical symbols with atomic numbers
        for z in znucl:
            symbol = chemical_symbols[int(z)]
            species_count[symbol] += 1
            if species_count[symbol] > 1:
                atom_type = symbol + str(species_count[symbol])
            else:
                atom_type = symbol
            atom_types.append(atom_type)
        # outcome: species_count & atom_types

        atom_labels = []
        if type(typat) is list and len(typat) > 1:
            for atom_index in range(natom[-1]):
                atom_labels.append(atom_types[typat[atom_index] - 1])
        elif type(typat) is int:
            atom_labels = atom_types
        sec_system.atom_labels = atom_labels


        # EIGENVALUES
        eig_file = dataset.get('EIG_file')

        if eig_file is None:
            logger.warning('Eigenvalue filename not found in mainfile')
        else:
            dirname = os.path.split(self.mainfile)[0]
            eig_fname = dirname + '/' + eig_file
            eig_found = os.path.isfile(eig_fname)
            if eig_found:
                self.parse_EIG_file(eig_fname, dataset, sscc, jdtset)
            else:
                logger.warning('Eigenvalue filename reported in mainfile was not found')

        # DENSITY OF STATES: DOS
        mainfile_base = (self.mainfile).split('.out')[0]
        dos_fname_multi = mainfile_base + f'o_DS{jdtset}_DOS'

        # consider also a job with *one* dataset, e.g., `<BASE>o_DOS`
        dos_fname_single = mainfile_base + 'o_DOS'


        if os.path.isfile(dos_fname_multi):
            dos_fname = dos_fname_multi
        elif os.path.isfile(dos_fname_single):
            dos_fname = dos_fname_single
        else:
            dos_fname = None
            logger.warning(f'DOS file missing for SSCC {jdtset}')

        # call
        if dos_fname is not None:
            logger.warning(f'DOS file found for SSCC {jdtset}, parsing ...')
            self.parse_DOS_file(dos_fname, dataset, sscc, jdtset)

    # end of parse_dataset()


##########################

    def parse_EIG_file(self, eig_file, dataset, sscc, sscc_idx):
        """
        Parse ABINIT _EIG file when eigenvalues are not fully reported in mainfile.

        Called from 'parse_dataset()'

        Args:
            eig_file: filename for eigenvalue file
            dataset: UnstructuredTextFileParser Object
            sscc : a section_single_configuration_calculation instance
            sscc_idx (int): current sscc (dataset) index
        """

        # In ABINIT, NOMAD`s 'sscc' are called 'datasets', and are numbered from one.
        # We pick up DOS file according to sscc (dataset) index.
        # BEWARE: the DOS output file can have different name patterns.
        # The only guarantee is that it will contain the dataset index
        # and that it will end with `_DOS`.
        #
        # logger.warning('parsing EIG file %s for SSCC # %d' %(eig_file, sscc_idx))

        # MAINFILE:
        occ_regex = r'END DATASET[\S\s]*?occ%s([\s\-0-9\.]+)occ' %(sscc_idx)
        quantities = [
            Quantity('occupations', occ_regex, repeats=False)
        ]
        parser_mainfile = UnstructuredTextFileParser(self.mainfile, quantities)
        occ = parser_mainfile.get('occupations')

        if occ is None:
            occ_regex = r'END DATASET[\S\s]*?occ([\s\-0-9\.]+)\w'
            quantities = [
                Quantity('occupations', occ_regex, repeats=False)
            ]
            parser_mainfile = UnstructuredTextFileParser(self.mainfile, quantities)
            occ = parser_mainfile.get('occupations')

        # _EIG FILE
        kpt_regex = r'kpt=\s*([\-0-9\. ]+)\s*\(reduced coord'
        band_regex = r'coord\)\s*([\-0-9\. ]+)\n'
        nband_regex = r'nband=\s*([0-9]+),'
        wtk_regex = r'wtk=\s*([\-0-9\.]+),'
        unit_regex = r'Eigenvalues \((\w+)\)'
        fermi_regex = r'\sFermi \(or HOMO\) energy \((?P<__unit>\w+)\) =\s*([\-0-9\.]+)'
        spinpol_regex = r'k points[\:\,]+([\w\s]+)' # `k points:` and `k points, SPIN UP:`


        def kpt_break(match_str):
            return match_str.strip().split()

        def spin_pol(match_str):
            if 'SPIN' in match_str:
                spinpol = 2
            else:
                spinpol = 1
            return spinpol

        eig_quantities = [
            Quantity('unit', unit_regex, repeats=False),
            Quantity('kpt_coords', kpt_regex, str_operation=kpt_break),
            Quantity('kpt_wtk', wtk_regex),
            Quantity('band_ene', band_regex),
            Quantity('nband', nband_regex),
            Quantity('fermi_ene', fermi_regex, repeats=False),
            Quantity('spin_channels', spinpol_regex, str_operation=spin_pol, repeats=False)
        ]

        parser = UnstructuredTextFileParser(eig_file, eig_quantities)

        unit = parser.get('unit')
        kpt_wtk = parser.get('kpt_wtk')
        nband = parser.get('nband')[0]
        band_ene = parser.get('band_ene', unit=unit)
        # fermi_ene = parser.get('fermi_ene', unit=unit)
        spin_channels = parser.get('spin_channels')

        eig_kpts = parser.get('kpt_coords')
        if spin_channels == 1:
            num_eig_kpt = len(eig_kpts)
        elif spin_channels == 2:
            num_eig_kpt = int(0.5 * len(eig_kpts))

        # SECTION EIGENVALUES: fill Archive metadata
        eigenval_sec = sscc.m_create(section_eigenvalues)

        if len(occ)==nband:
            # if ABINIT's `occopt==1`, then all kpts have the same occ
            # hence, we need to mirror array, so that later reshape succedes
            occ = np.repeat(occ, num_eig_kpt)

        eigenval_sec.eigenvalues_kind = 'electronic'
        eigenval_sec.number_of_eigenvalues_kpoints = num_eig_kpt
        eigenval_sec.eigenvalues_kpoints_weights = kpt_wtk
        eigenval_sec.eigenvalues_kpoints = eig_kpts
        eigenval_sec.eigenvalues_occupation = np.reshape(occ, (spin_channels, num_eig_kpt,nband))
        eigenval_sec.eigenvalues_values = np.reshape(band_ene, (spin_channels, num_eig_kpt,nband))
        eigenval_sec.number_of_eigenvalues = nband
    # end of parse_EIG_file()

    def parse_DOS_file(self, dos_file, dataset, sscc, sscc_idx):
        """
        Parse ABINIT's _DOS file.

        Called from 'parse_dataset()'

        Args:
            dos_file: filename for density-of-states file
            dataset: UnstructuredTextFileParser Object
            sscc : a section_single_configuration_calculation instance
            sscc_idx (int): current sscc (dataset) index
        """

        try:
            with open(dos_file, 'r') as textfile:
                body = textfile.read()
        except FileNotFoundError:
            logger.warning(f'File not found: {dos_file}')
        except Exception as err:
            logger.error(f'Exception on {__file__}', exc_info=err)

        # DOS file: identify energy units
        regex = r'\s*at\s*(?P<num_dos_values>\d*)\s*energies\s*\(in\s*(?P<energy_unit>\w*)\)'
        match = re.search(regex, body, re.MULTILINE)
        if match:
            num_dos_values = int(match.group('num_dos_values'))
            energy_unit = match.group('energy_unit')
            if energy_unit == 'Hartree':
                units_dos_file = ureg.a_u_energy
            elif energy_unit == 'eV':
                units_dos_file = ureg.eV

        # DOS file: pick up Fermi energy
        match = re.search(
            r'^#\s*Fermi energy :\s*(?P<fermi_energy>[-+]*\d*\.\d*)', body, re.MULTILINE)
        if match:
            # `fermiFU`: energy_fermi with `file` units (eV or Hartree)
            fermiFU = float(match.group('fermi_energy')) * units_dos_file
            # normalizer expects Joules
            fermi_energy_J = fermiFU.to(ureg.J)
            sscc.energy_reference_fermi = np.array(
                    fermi_energy_J.magnitude, ndmin=1)

        # DOS file: open it again, this time directly to a Numpy array
        try:
            dos_data = np.genfromtxt(dos_file)
        except FileNotFoundError:
            logger.warning(f'File not found: {dos_file}')
        except Exception as err:
            logger.error(f'Exception on {__file__}', exc_info=err)

        # Slice `dos_data` according to `num_dos_values`. Doing so way we treat
        # correctly the number of spin levels
        if dos_data.shape[0] == num_dos_values:
            spin_treat = False
        else:
            spin_treat = True

        dos_energies_Joules = (
            dos_data[:num_dos_values, 0] * units_dos_file).to(ureg.J)
        dos_values = np.zeros((2, num_dos_values))
        dos_values_integrated = np.zeros((2, num_dos_values))
        if spin_treat:
            # start till num_dos_values
            dos_values[0] = dos_data[:num_dos_values, 1]
            # num_dos_values till end
            dos_values[1] = dos_data[num_dos_values:, 1]
            # likewise
            dos_values_integrated[0] = dos_data[:num_dos_values, 2]
            dos_values_integrated[1] = dos_data[num_dos_values:, 2]
        else:
            dos_values[0] = dos_data[:num_dos_values, 1]
            dos_values[1] = dos_data[:num_dos_values, 1]
            dos_values_integrated[0] = dos_data[:num_dos_values, 2]
            dos_values_integrated[1] = dos_data[:num_dos_values, 2]

        # NOMAD metainfo needs dos_values (A) without physical units,
        # (B) without unit-cell normalization, and (C) without Fermi-energy shift
        # In ABINIT
        #   - DOS units are (electrons/Hartree/cell) and
        #   - integrated DOS are in (in electrons/cell)
        #   - `_DOS` file has dos_values without Fermi shift,
        #   - `_DOS` file uses energies in Hartree, regardless of the value
        #     of ABINIT's variable `enunit` (energy units for bandstructures)

        # Retrieve  unit cell volume.
        # Original value was in 'bohr**3', but the Archive stores it in 'meter**3'
        # hence we need to convert it back to bohrs**3
        unit_cell_vol_bohr3 = sscc.x_abinit_unit_cell_volume.to('bohr**3')

        dos_values = dos_values * unit_cell_vol_bohr3.magnitude
        dos_values_integrated = dos_values_integrated * unit_cell_vol_bohr3.magnitude

        # SECTION DOS: creation and filling
        dos_sec = sscc.m_create(section_dos)
        dos_sec.dos_kind = 'electronic'
        dos_sec.number_of_dos_values = dos_values.shape[0]
        dos_sec.dos_energies = dos_energies_Joules

        dos_sec.dos_values = dos_values
        dos_sec.dos_integrated_values = dos_values_integrated
    # end of parse_DOS_file()

    # = = = = = = = = = =
    def parse(self):
        root_out = self.out_parser.get('root_out')
        ndtset = self.out_parser.get('ndtset')[0]
        logger.warning(f'Root for output filenames: {root_out}')
        logger.warning(f'Number of datasets found: {ndtset}')

        # SECTION CREATION
        sec_run = self.archive.m_create(section_run)

        # SECTION METHOD
        self.parse_method()

        # ARCHIVE FILLING
        sec_run.program_name = 'abinit'
        sec_run.program_version = self.out_parser.get('program_version')
        sec_run.program_basis_set_type = 'plane waves'

        # Energy Cutoff
        sec_basis_set_cell_dependent = sec_run.m_create(section_basis_set_cell_dependent)
        sec_basis_set_cell_dependent.basis_set_cell_dependent_kind = 'plane_waves'
        energy_cutoff = self.out_parser.get('ecut', unit='hartree')
        if energy_cutoff is not None:
            sec_basis_set_cell_dependent.basis_set_planewave_cutoff = energy_cutoff
            #
            sec_basis_set_cell_dependent.basis_set_cell_dependent_name = "PW_%s" % (energy_cutoff.to('rydberg').magnitude)

        # Always pick first entry of `ndtset`: this sets the num of SSCC's found.
        # Some quantities will need to be trim off  up to /maximum/ ndtset occurrences
        ndtset = self.out_parser.get('ndtset')[0]

        getwfk = self.out_parser.get('getwfk')
        if getwfk is not None:
            getwfk = getwfk[:ndtset] # TODO: store archive: method

        getden = self.out_parser.get('getden')
        if getden is not None:
            getden = getden[:ndtset] # TODO: store archive: method

        nkpt = self.out_parser.get('nkpt')[:ndtset]
        amu = self.out_parser.get('amu')
        symrel = self.out_parser.get('symrel') # formerly at x_abinit_section_dataset.x_abinit_section_input

        # kpoints: one entry per dataset
        kpt_list = self.out_parser.get('kpt')

        # for ii, kpt in enumerate(kpt_list):
        #    print(f'kpt {ii+1})\n ',   kpt)
        #    print('----')

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        # SUB-PARSING: TAIL REPORT
        #print('# TAIL REPORT #######################\n')
        # tail_report = self.out_parser.get('tail_report')
        #print(type(tail_report))
        # self.parse_tail(tail_report)

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        # SUB-PARSING: DATASETS
        dataset_set = self.out_parser.get('dataset')

        for idx, dataset in enumerate(dataset_set):
            self.parse_dataset(dataset, idx+1)

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        # CROSS-REFERENCES
        # MARKUS PROPOSAL: first do all parsing, then do all cross-references.

        # gIndex = backend.openSection("section_method_basis_set")
        # backend.addValue( "mapping_section_method_basis_set_cell_associated", self.basisGIndex)

        # Previous two lines from old parser become the next line in the new infraestructure
        sec_run.section_method[0].section_method_basis_set[0].mapping_section_method_basis_set_cell_associated = sec_run.section_basis_set_cell_dependent[0]
        #print(sec_run.section_method[0].section_method_basis_set[0].m_def.all_quantities)

    # = = = = = = =
    # end of parse()

