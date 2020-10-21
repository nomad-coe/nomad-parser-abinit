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

from .metainfo import m_env
from nomad.parsing.parser import FairdiParser
from abinitparser.parser2 import AbinitParserInterface



class AbinitParser(FairdiParser):
    def __init__(self):
        super().__init__(
        name='parsers/abinit', code_name='ABINIT', code_homepage='https://www.abinit.org/',
        mainfile_contents_re=(r'^\n*\.Version\s*[0-9.]*\s*of ABINIT\s*'))

    def parse(self, filepath, archive, logger=None):
        self._metainfo_env = m_env

        parser = AbinitParserInterface(filepath, archive, logger)

        parser.parse()
