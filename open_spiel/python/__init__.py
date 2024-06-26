# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Open Spiel Python API."""

from typing import Dict, Union

# A type provided for PyType hints. Added after the discussion in
# https://github.com/google-deepmind/open_spiel/issues/1224.
GameParameter = Union[int, float, str, bool, Dict[str, 'GameParameter']]

