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

from nomad.datamodel.metainfo.public import (
    section_run, section_system, section_basis_set_cell_dependent,
    section_dos, section_k_band, section_k_band_segment, section_eigenvalues,
    section_single_configuration_calculation)

from nomad.parsing.text_parser import Quantity, UnstructuredTextFileParser

class AbinitOutputParser(UnstructuredTextFileParser):
    """Parser of an ABINIT file"""
    def __init__(self, mainfile, logger):
        super().__init__(mainfile, None, logger)


    def init_quantities(self):
        # def string_to_vars(string):
        #     lines = string.strip().split('\n')
        #     outvars = dict()
        #     for line in lines:
        #         split = line.split()
        #         outvars[split[0]] = split[1:]

        outvars = {}
        float_re = r'([\d\.E\+\-]+)\s*' # (?P<__unit>\w*)
        float_re2 = r'([\d\.E\+\-]+)' # (?P<__unit>\w*)
        array_re = r'([\d\.E\+\-\s]+)'

        self._quantities = []
        for key in ['amu', 'diemac', 'dosdeltae', 'ecut', 'kptrlen1', 'kptrlen2', 'znucl']:
            repat = r'%s\s*%s' %(key, float_re)
            self._quantities.append(Quantity(key, repat, repeats=False))

        print(self._quantities)

        # arrays
        for key in ['acell', 'rprim', 'tnons', 'symrel', 'xangst', 'xcart', 'xred']:
            repat = r'%s\s*%s' %(key, array_re)
            self._quantities.append(Quantity(key, repat, repeats=False))

        #############
        def string_to_kpt2(string):
            print(string)
            val = string.strip().split('kpt')
            kpoints = []
            for item in val:
                item = item.split()
                if len(item) < 3:
                    continue
                kpoints.append(np.reshape(item, (len(item)//3, 3)))
            return kpoints

        def string_to_kpt(string):
            print(string)
            val = string.strip().split('kpt')
            kpoints = []
            for item in val:
                item = item.split()
                if len(item) < 3:
                    continue
                item = item[1:] if item[0].isdecimal() else item

                kpoints.append(np.reshape(item, (len(item)//3, 3)))
            return kpoints

        def string_to_ener(string):
            print(string)
            val = [v.split('=') for v in string.strip().split('\n')]
            return {v[0].strip().lstrip('>'):pint.Quantity(float(v[1]),'hartree') for v in val}




        self._quantities.append(Quantity('kpt', r'-outvars: echo values of preprocessed input variables --------[\s\S]*?(kpt[kpt\d\.E\+\-\s]+)',str_operation=string_to_kpt, repeats=False))

        self._quantities.append(Quantity('energies', r'Components of total free energy \(in Hartree\) :([\s\S]*?Etotal=\s*[\d\.E\+\-]+)', str_operation=string_to_ener, repeats=True))

# Components of total free energy (in Hartree) :

#     Kinetic energy  =  3.06072795183464E+00
#     Hartree energy  =  5.42787238645061E-01
#     XC energy       = -3.54910501798730E+00
#     Ewald energy    = -8.46648022654903E+00
#     PspCore energy  =  8.69853998687012E-02
#     Loc. psp. energy= -2.41963695123264E+00
#     NL   psp  energy=  1.87849770943371E+00
#     >>>>>>>>> Etotal= -8.86622389598685E+00

#  Other information on the energy :
#     Total energy(eV)= -2.41262221822403E+02 ; Band energy (Ha)=   2.6932045268E-01



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

 #acell      1.0180000000E+01  1.0180000000E+01  1.0180000000E+01 Bohr

class AbinitParserInterface:
    """Write to the Archive"""
    def __init__(self, mainfile, archive, logger):
        self.mainfile = mainfile
        self.archive = archive
        self.logger = logger
        self.out_parser = AbinitOutputParser(mainfile, logger)

    def parse(self):
        # =========
        sec_run = self.archive.m_create(section_run)
        sec_run.program_name = 'abinit'
        sec_run.program_version = self.out_parser.get('program_version')

        # =========
        # MARKUS: how to create s subsection?

        sec_basis_set_cell_dependent = sec_run.m_create(section_basis_set_cell_dependent) # MARKUS:

        # store
        basis_set_planewave_cutoff = self.out_parser.get('energy_cutoff')


        # =========
        sec_system = sec_run.m_create(section_system)

        # lattice_constant = self.out_parser.get('lattice_constant')[-1]
        # lattice_vectors = self.out_parser.get('lattice_vectors')[-1]
        # lattice_vectors = np.reshape(lattice_vectors, (3,3))
        # lattice_vectors_dim = np.zeros_like(lattice_vectors)

        # apply 'acell' factor
        # for ii in range(3):
        #     lattice_vectors_dim[ii] = lattice_vectors[ii] * lattice_constant[ii]
        # sec_system.lattice_vectors = lattice_vectors_dim


        # =========


        print('jdtset (num of SSCC`s) ' , self.out_parser.get('num_sscc'))
        print('getden ' , self.out_parser.get('getden'))

        print(self.out_parser.get('amu'))
        print('acell', self.out_parser.get('acell'))
        print(self.out_parser.get('rprim'))

        #print('kpt', self.out_parser.get('kpt'))

        kpts = self.out_parser.get('kpt')
        for kpt in kpts:
            print(kpt)
            print('----')

        print('energies', len(self.out_parser.get('energies')))
        energy_list = self.out_parser.get('energies')

        energy_map = {}
        energy_map['Kinetic energy'] = 'electronic_kinetic_energy'



        for ener in energy_list:
            sscc = sec_run.m_create(section_single_configuration_calculation)
            for key, val in ener.items():
                if key in energy_map:
                    setattr(sscc, energy_map[key], val)





