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
import numpy as np
from nomad.units import ureg

from nomad.datamodel.metainfo.public import (
    section_run, section_system, section_basis_set_cell_dependent,
    section_dos, section_k_band, section_k_band_segment, section_eigenvalues,
    section_single_configuration_calculation)

from nomad.parsing.text_parser_new import Quantity, UnstructuredTextFileParser

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
        #     Quantity('lattice_vectors', r'rprim\s*([\d\.E\+\-\s]+)', repeats=True),
        #     Quantity('lattice_constant', r'acell\s*([\d\.E\+\-\s]+)', repeats=True),
        #     Quantity('energy_cutoff', r'ecut\s*([\d\.E\+\-]+)', repeats=False),
        #     Quantity('getden', r'(getden\d\s*[\d]+)', repeats=True),
        #     Quantity('getwfk', r'(getwfk\d\s*[\d]+)', repeats=True),
        #     Quantity('iscf', r'(iscf\d\s*[\d]+)', repeats=True),
        #     Quantity('istwfk', r'(istwfk\d\s*[\d]+)', repeats=True),
        #     Quantity('outvars_block', r'-outvars: echo values of preprocessed input variables --------([\s\S]+?)={80}', repeats=False),
        #]

        #############
        def string_to_ener(string):
            val = [v.split('=') for v in string.strip().split('\n')]
            for v in val:
                v[0] = v[0].strip().lstrip('>').strip()
            return {v[0]:pint.Quantity(float(v[1]),'hartree') for v in val}


        def string_to_kpt_OBSOLETE(string):
            #print(string)
            val = string.strip().split('kpt')
            kpoints = []
            for item in val:
                item = item.split()
                if len(item) < 3:
                    continue
                kpoints.append(np.reshape(item, (len(item)//3, 3)))
            return kpoints

        def string_to_kpt(string):
            # print('kpt', string)
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
            # first index: matrix handle
            return np.reshape(symrel_list, (-1,3,3))

        outvars = {}
        # Common regex's
        int_re = r'([\-\d]+)'
        float_re = r'([\d\.E\+\-]+)\s*' # (?P<__unit>\w*)
        float_re2 = r'([\d\.E\+\-]+)'
        array_re = r'([\d\.E\+\-\s]+)'




        # INTEGER Variables
        int_list = ['ndtset', 'fftalg', 'getden', 'getwfk', 'iscf',
            'mkmem', 'natom', 'nband', 'nbdbuf', 'nkpt',
            'snstep', 'nsym', 'ntypat', 'prtdos', 'spgroup']

        for key in int_list:
            # added '[\d]' to account for possible index: nkpt1, nkpt2, ...
            regex = r'%s[\d]*\s*%s' %(key, int_re)
            self._quantities.append(Quantity(key, regex, repeats=True))

        # FLOAT Variables
        float_list = ['amu', 'diemac', 'dosdeltae', 'ecut', 'kptrlen', 'znucl']
        for key in float_list:
            regex = r'%s[\d]*\s*%s' %(key, float_re)
            self._quantities.append(Quantity(key, regex, repeats=False))

        # ARRAY Variables
        for key in ['acell', 'rprim', 'tnons', 'symrel', 'xangst', 'xcart', 'xred']:
            repat = r'%s\s*%s' %(key, array_re)
            self._quantities.append(Quantity(key, repat, repeats=False))


#

        # KPTs
        self._quantities.append(Quantity('kpt', r'-outvars: echo values of preprocessed input variables --------[\s\S]*?(kpt[kpt\d\.E\+\-\s]+)',str_operation=string_to_kpt, repeats=False))
        self._quantities.append(Quantity('symrel', r'-outvars: echo values of preprocessed input variables --------[\s\S]*?(symrel[symrel\d\-\s]+)',str_operation=string_to_symrel, repeats=False))
        # ENERGIES
        self._quantities.append(Quantity('energies', r'Components of total free energy \(in Hartree\) :([\s\S]*?Etotal=\s*[\d\.E\+\-]+)', str_operation=string_to_ener, repeats=True))

        # DATASETS SUBPARSING
        def str_to_scf_ener(string):
            # pick third column if line starts with 'ETOT'
            val = [v.split()[2] for v in string.strip().split('\n') if 'ETOT' in v]
            return val

        def str_to_force(string):
            val = string.strip().split()
            val = np.reshape(val, (-1,4))
            return val[:,1:] # skip first column (it's an index)

        dataset_quantity = Quantity(
            'dataset', r'== DATASET\s*\d+([\s\S]*?)(?:== DATASET\s*\d*|== END DATASET)',
            repeats=True,
            sub_parser=UnstructuredTextFileParser(quantities=[
                Quantity(
                    'scf_energies',
                    r'\n *iter([\s\S]*?)converged', str_operation=str_to_scf_ener,
                    repeats=False),
                Quantity(
                    'stress',
                    r'\s* Cartesian components of stress tensor \(hartree[/]bohr.3\)'
                    r'\s*sigma\(\d \d\)=\s*([\+\-\d\.Ee]+)'
                    r'\s*sigma\(\d \d\)=\s*([\+\-\d\.Ee]+)'
                    r'\s*sigma\(\d \d\)=\s*([\+\-\d\.Ee]+)'
                    r'\s*sigma\(\d \d\)=\s*([\+\-\d\.Ee]+)'
                    r'\s*sigma\(\d \d\)=\s*([\+\-\d\.Ee]+)'
                    r'\s*sigma\(\d \d\)=\s*([\+\-\d\.Ee]+)',
                    repeats=False, dtype=float, shape=(3,2), unit='hartree/bohr**3'),
                Quantity(
                    'ucvol',
                    r'\s*Unit cell volume ucvol=\s*([\+\-\d\.Ee]+)', dtype=float,
                    unit='bohr**3', repeats=False),
                Quantity(
                    'EIG_file',
                    r'\s*prteigrs : about to open file\s*([\S]+)'),
                Quantity(
                    'forces_SCF',
                    r'\s*cartesian forces \(hartree/bohr\) at end:\s*([\+\-\d\.\n ]+)',
                    str_operation=str_to_force, dtype=float, unit='hartree/bohr')
                    ] ))

        self._quantities.append(dataset_quantity)



class AbinitParserInterface:
    """Class to write to NOMAD's Archive"""
    def __init__(self, mainfile, archive, logger):
        self.mainfile = mainfile
        self.archive = archive
        self.logger = logger
        self.out_parser = AbinitOutputParser(mainfile, logger)

    def parse(self):
        # Create Sections
        sec_run = self.archive.m_create(section_run)
        sec_system = sec_run.m_create(section_system)
        sec_basis_set_cell_dependent = sec_run.m_create(section_basis_set_cell_dependent)

        # Fill in
        sec_run.program_name = 'abinit'
        sec_run.program_version = self.out_parser.get('program_version')


        # =========
        # Energy Cutoff
        energy_cutoff = self.out_parser.get('ecut')
        sec_basis_set_cell_dependent.basis_set_planewave_cutoff = energy_cutoff

        # =========
        # Lattice vectors and lattice parameter
        lattice_constant = self.out_parser.get('acell')
        rprim = np.reshape(self.out_parser.get('rprim'), (3,3))
        lattice_vectors = np.zeros_like(rprim)

        # apply 'acell' factor
        for ii in range(3):
            lattice_vectors[ii] = rprim[ii] * lattice_constant[ii]
        sec_system.lattice_vectors = lattice_vectors
        # =========

        # Always pick first entry: this sets the num of SSCC's found
        # then trim off all quantities at /maximum/ ndtset occurrences
        ndtset = self.out_parser.get('ndtset')[0]
        getden = self.out_parser.get('getden')[:ndtset]
        nkpt = self.out_parser.get('nkpt')[:ndtset]
        amu = self.out_parser.get('amu')
        symrel = self.out_parser.get('symrel')

        natom = self.out_parser.get('natom')[:ndtset]
        znucl = self.out_parser.get('znucl')
        sec_system.atom_atom_number = znucl

        xcart =  np.reshape(self.out_parser.get('xcart'), (-1,3))
        sec_system.atom_positions = xcart * ureg.a_u_length



        # kpoints: one entry per dataset
        kpt_list = self.out_parser.get('kpt')

        for ii, kpt in enumerate(kpt_list):
            print(f'kpt {ii+1})\n ',   kpt)
            print('----')

        # ENERGIES
        energy_list = self.out_parser.get('energies')

        # map ABINIT to NOMAD Metadata
        energy_map = {}
        energy_map['Kinetic energy'] = 'electronic_kinetic_energy'
        energy_map['Hartree energy'] = 'energy_correction_hartree'
        energy_map['XC energy'] = 'energy_XC'
        energy_map['Etotal'] = 'energy_total'
        energy_map['PspCore energy'] = 'x_abinit_energy_psp_core'
        energy_map['EIG_file'] = 'x_abinit_energy_ewald'
        energy_map['Loc. psp. energy'] = 'x_abinit_energy_psp_local'
        energy_map['NL   psp  energy'] = 'x_abinit_energy_psp_nonlocal'


        for ener in energy_list:
            print('ener', ener)
            sscc = sec_run.m_create(section_single_configuration_calculation)
            for key, val in ener.items():
                if key in energy_map:
                    print('key', key)
                    metainfokey = energy_map[key]
                    if metainfokey.startswith('x_abinit'):
                        val = val.to('joule').magnitude
                    setattr(sscc, energy_map[key], val) #  Hartree

        #sscc = sec_run.m_create(section_single_configuration_calculation)
        #sscc = sec_run.m_create(section_single_configuration_calculation)

        print('ndtset (num of SSCC`s)', ndtset)
        print('getden ', getden)
        print('nkpt', nkpt)
        print('amu ', amu)
        print('nkpt', nkpt)

        print('lattice_constant', lattice_constant)
        print('rprim', rprim)
        print('lattice_vectors', lattice_vectors)

        print('xcart', xcart)
        print(natom, znucl)


        datasets = self.out_parser.get('dataset')
        for dataset in datasets:
            print('\n\n\nDATASET')
            print('scf_energies', dataset.get('scf_energies'))
            print('\nstress', dataset.get('stress'))
            print('\nucvol', dataset.get('ucvol'))
            print('\nEIG_file', dataset.get('EIG_file'))
            print('\nforces_SCF', dataset.get('forces_SCF'))








# xred:
# atom_positions_primitive
# atom_positions_std