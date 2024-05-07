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

"""Example to traverse an entire game tree."""

from absl import app
from absl import flags

import numpy as np
from open_spiel.python import games  # pylint: disable=unused-import
from open_spiel.python.bots import human
import pyspiel



def play_game(state: pyspiel.State,
              bots: list[pyspiel.Bot]):
  """Play the game via console."""

  while not state.is_terminal():
    print(f"State: \n{state}\n")
    if state.is_chance_node():
      outcomes = state.chance_outcomes()
      action_list, prob_list = zip(*outcomes)
      outcome = np.random.choice(action_list, p=prob_list)
      print(f"Chance chose: {outcome} ({state.action_to_string(outcome)})")
      state.apply_action(outcome)
    else:
      player = state.current_player()
      action = bots[player].step(state)
      print(f"Chose action: {action} ({state.action_to_string(action)})")
      state.apply_action(action)

  print("\n-=- Game over -=-\n")
  print(f"Terminal state:\n{state}")
  print(f"Returns: {state.returns()}")
  return


def main(_):
  game = pyspiel.load_game("quoridor(board_size=4,players=4,relative_moves=true)")
  state = game.new_initial_state()
  bots = []
  bots.append(human.HumanBot())
  bots.append(human.HumanBot())
  bots.append(human.HumanBot())
  bots.append(human.HumanBot())
  play_game(state, bots)


if __name__ == "__main__":
  app.run(main)
