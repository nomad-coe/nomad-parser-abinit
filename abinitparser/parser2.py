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

import pint
import logging
import numpy as np
from nomad.units import ureg
from ase.data import chemical_symbols
from nomad.parsing.text_parser import Quantity, UnstructuredTextFileParser

from abinitparser.AbinitXC import ABINIT_NATIVE_IXC, ABINIT_LIBXC_IXC

from nomad.datamodel.metainfo.public import (
    section_run, section_system,  section_method, section_basis_set_cell_dependent,
    section_dos, section_k_band, section_k_band_segment, section_eigenvalues,
    section_single_configuration_calculation, section_XC_functionals,
    section_symmetry
    )

from nomad.datamodel.metainfo.common import (
    section_method_basis_set)




logger = logging.getLogger("nomad.ABINITParser")


class AbinitOutputParser(UnstructuredTextFileParser):
    """Parser of an ABINIT file"""
    def __init__(self, mainfile, logger):
        super().__init__(mainfile, None, logger)


    def init_quantities(self):
        self._quantities = []
        self._quantities.append(Quantity('program_version', r'\.Version\s*([\d\.]+)', repeats=False),)

        # self._quantities = [
        #     Quantity('num_sscc', r'jdtset\s*([\d\s]+)', repeats=False),
        #     Quantity('program_version', r'\.Version\s*([\d\.]+)', repeats=False),
        #     Quantity('lattice_constant', r'acell\s*([\d\.E\+\-\s]+)', repeats=True), # not in metadata
        #     Quantity('istwfk', r'(istwfk\d\s*[\d]+)', repeats=True),
        #     Quantity('outvars_block', r'-outvars: echo values of preprocessed input variables --------([\s\S]+?)={80}', repeats=False),
        #]

        #############
        def string_to_ener(string):
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
            'amu', 'diemac', 'dosdeltae', 'ecut', 'toldfe', 'tolwfr', 'tsmear',
            'kptrlen']
        for key in float_list:
            regex = r'%s[\d]*\s+%s' %(key, float_re)
            self._quantities.append(Quantity(key, regex, repeats=True))

        # ARRAY Variables
        array_list =[
            'acell', 'rprim', 'tnons', 'symrel', 'istwfk','typat', 'xangst',
            'xcart', 'xred', 'znucl']
        for key in array_list:
            #repat = r'%s\s*%s' %(key, array_re)
            repat = r'%s[\d]*\s+%s' %(key, array_re)
            self._quantities.append(Quantity(key, repat, repeats=False))

        # SPECIAL CASES
        #self._quantities.append(Quantity('root_out',r'- root for output files ->\s*([\s]+)', repeats=False)

        self._quantities.append(Quantity('kpt', r'-outvars: echo values of preprocessed input variables --------[\s\S]*?(kpt[kpt\d\.E\+\-\s]+)',str_operation=string_to_kpt, repeats=False))
        self._quantities.append(Quantity('symrel', r'-outvars: echo values of preprocessed input variables --------[\s\S]*?(symrel[symrel\d\-\s]+)',str_operation=string_to_symrel, repeats=False))
        self._quantities.append(Quantity('nsppol', r'\s*nsppol\s*=\s*(\d)', repeats=False))
        self._quantities.append(Quantity('occopt', r'\s*occopt\s*=\s*(\d)', repeats=True))
        self._quantities.append(Quantity('ixc', r'[\s\S]*ixc=([\-\d]*)|ixc\s*([\-\d]*)', repeats=True))

        # SUBPARSING for DATASETS
        def str_to_scf_ener(string):
            # pick third column if line starts with 'ETOT'
            val = [v.split()[2] for v in string.strip().split('\n') if 'ETOT' in v]
            return val

        def str_to_force(string):
            val = string.strip().split()
            val = np.reshape(val, (-1,4))
            return val[:, 1:]  # skip first column (it's an atom index)

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

        dataset_quantity = Quantity(
            'dataset',r' Exchange-correlation([\s\S]*?)(?:== DATASET  \d+|== END DATASET)',
            # - - - - - -
            #'dataset', r'== DATASET\s*\d+([\s\S]*?)(?:== DATASET  \d+|== END DATASET\(S\) )',
            repeats=True,
            sub_parser=UnstructuredTextFileParser(quantities=[
                Quantity(
                    'scf_energies', # TODO: archive
                    r'\n *iter([\s\S]*?)converged', str_operation=str_to_scf_ener,
                    repeats=False),
                Quantity(
                    'energies',
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
                    'ucvol', # TODO: archive
                    r'\s*Unit cell volume ucvol=\s*([\+\-\d\.Ee]+)', dtype=float,
                    unit='bohr**3', repeats=False),
                Quantity(
                    'EIG_file',# TODO: archive
                    r'\s*prteigrs : about to open file\s*([\w]+)', repeats=False),
                Quantity(
                    'forces_SCF', # TODO: archive
                    r'\s*cartesian forces \(hartree/bohr\) at end:\s*([\+\-\d\.\n ]+)',
                    str_operation=str_to_force, dtype=float, unit='hartree/bohr'),
                Quantity(
                    'lattice_vec',
                    r'R\(\d\)=\s*([ \d\.Ed\-]+)', repeats=True),
                Quantity(
                    'xcart_dataset',
                    r'\s*cartesian coordinates \(angstrom\) at end:\s*\n([\-\d\. \n]*)',
                    str_operation=str_to_xcart, unit='angstrom', repeats=False
                )
                    ]))

        self._quantities.append(dataset_quantity)


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
        sec_method.scf_threshold_energy_change = self.out_parser.get('toldfe', unit='hartree')
        sec_method.self_interaction_correction_method = "" # empty str in original parser

        # Method tolerances
        tolwfr = self.out_parser.get('tolwfr')
        toldfe = self.out_parser.get('toldfe')
        sec_method.scf_threshold_energy_change = pint.Quantity(toldfe, 'hartree')
        sec_method.x_abinit_tolwfr = tolwfr
        sec_method.x_abinit_istwfk = self.out_parser.get('istwfk')


        # SMEAR KIND
        occopt = self.out_parser.get('occopt')[0]
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


    def parse_dataset(self, dataset):
        """ Parses an UnstructuredTextFileParser object and stores relevant data
        into NOMAD's Archive

        Args:
            dataset: UnstructuredTextFileParser Object
        """
        # sec_run:    a single call to a program
        # sec_system: physical properties that define a system
        # sec_scc:    values computed during a single config calc,
        #             for a given method and a given system

        sec_run = self.archive.section_run[-1]
        sec_system = sec_run.m_create(section_system)
        sscc = sec_run.m_create(section_single_configuration_calculation)
        ndtset = self.out_parser.get('ndtset')[0]

        spgroup = self.out_parser.get('spgroup')
        sec_symm = sec_system.m_create(section_symmetry)
        sec_symm.space_group_number = spgroup

        sec_system.configuration_periodic_dimensions = np.array([True, True, True])
        sec_system.number_of_atoms = int(self.out_parser.get('natom')[-1])

        lat_vec = dataset.get('lattice_vec', unit='bohr')
        sec_system.lattice_vectors = lat_vec
        sec_system.simulation_cell = lat_vec

        # ENERGIES
        energies = dataset.get('energies',{})
        for key, val in energies.items():
            if key in self._energy_map:
                metainfokey = self._energy_map[key]
                if metainfokey.startswith('x_abinit'):
                    val = val.to('joule').magnitude
                setattr(sscc, self._energy_map[key], val) #  Hartree

        stress = dataset.get('stress')
        if stress is not None:
            sscc.stress_tensor = stress

        # ATOM POSITIONS
        xcart_ds = dataset.get('xcart_dataset')
        if xcart_ds is not None:
            sec_system.atom_positions = xcart_ds # unit was attached during parsing
        else:
            logger.error("Atom positions within dataset are unavailable")

        # ATOM LABELS
        znucl = self.out_parser.get('znucl')
        natom = self.out_parser.get('natom')[:ndtset]
        ntypat = self.out_parser.get('ntypat')
        typat = self.out_parser.get('typat')

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
        for atom_index in range(natom[-1]):
            atom_labels.append(atom_types[typat[atom_index]-1])
        sec_system.atom_labels = atom_labels


    def parse(self):
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

        print('sec_basis_set_cell_dependent', sec_basis_set_cell_dependent)

        # Always pick first entry of `ndtset`: this sets the num of SSCC's found.
        # Then trim off all quantities at /maximum/ ndtset occurrences
        ndtset = self.out_parser.get('ndtset')[0]
        iscf = np.array(self.out_parser.get('iscf')[:ndtset], dtype=int)
        getwfk = self.out_parser.get('getwfk')[:ndtset]
        getden = self.out_parser.get('getden')[:ndtset]
        nkpt = self.out_parser.get('nkpt')[:ndtset]
        amu = self.out_parser.get('amu')
        symrel = self.out_parser.get('symrel') # formerly at x_abinit_section_dataset.x_abinit_section_input



        # sec_system.atom_atom_number = znucl # FIXME: needs generalization

        print('XCART:\n', self.out_parser.get('xcart'))
        xcart =  np.reshape(self.out_parser.get('xcart'), (-1,3))


        # kpoints: one entry per dataset
        kpt_list = self.out_parser.get('kpt')

        #for ii, kpt in enumerate(kpt_list):
        #    print(f'kpt {ii+1})\n ',   kpt)
        #    print('----')

        if 1==0:
            print('ndtset (num of SSCC`s)', ndtset)
            print('getden ', getden)
            print('nkpt', nkpt)
            print('amu ', amu)
            print('nkpt', nkpt)

            print('lattice_constant', lattice_constant)
            print('rprim', rprim)


            print('xcart', xcart)
            print(natom, znucl)

        # DATASET SUB-PARSING
        dataset_set = self.out_parser.get('dataset')

        for dataset in dataset_set:
            self.parse_dataset(dataset)


        # if [False,True][1]:
        #     for jdtset, dataset in enumerate(dataset_set):
        #         print('\n\n\nDATASET', jdtset)
        #         print('scf_energies', dataset.get('scf_energies'))
        #         print('energies', dataset.get('energies'))
        #         print('\nstress', dataset.get('stress'))
        #         print('\nucvol', dataset.get('ucvol'))
        #         print('\nEIG_file', dataset.get('EIG_file'))
        #         print('\nforces_SCF', dataset.get('forces_SCF'))

        # print('dataset_set', len(dataset_set), dataset_set)


        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        # CROSS-REFERENCES
        # MARKUS PROPOSAL: first do all parsing, then do all cross-references.

        # gIndex = backend.openSection("section_method_basis_set")
        # backend.addValue( "mapping_section_method_basis_set_cell_associated", self.basisGIndex)

        # Previous two lines from old parser become the next line in the new infraestructure
        sec_run.section_method[0].section_method_basis_set[0].mapping_section_method_basis_set_cell_associated = sec_run.section_basis_set_cell_dependent[0]
        #print(sec_run.section_method[0].section_method_basis_set[0].m_def.all_quantities)


