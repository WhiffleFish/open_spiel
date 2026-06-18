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

"""Tests for the game-specific functions for euchre."""


from absl.testing import absltest

import pyspiel
euchre = pyspiel.euchre


class GamesEuchreTest(absltest.TestCase):

  def _state_at_bidding(self):
    state = pyspiel.load_game('euchre').new_initial_state()
    state.apply_action(0)
    while state.is_chance_node():
      state.apply_action(state.chance_outcomes()[0][0])
    return state

  def _state_at_go_alone(self):
    state = self._state_at_bidding()
    call_action = next(
        action for action in state.legal_actions()
        if action != euchre.PASS_ACTION)
    state.apply_action(call_action)
    if state.current_phase() == euchre.Phase.DISCARD:
      state.apply_action(state.legal_actions()[0])
    self.assertEqual(state.current_phase(), euchre.Phase.GO_ALONE)
    return state

  def test_bindings(self):
    self.assertEqual(euchre.JACK_RANK, 2)
    self.assertEqual(euchre.NUM_SUITS, 4)
    self.assertEqual(euchre.NUM_CARDS_PER_SUIT, 6)
    self.assertEqual(euchre.NUM_CARDS, 24)
    self.assertEqual(euchre.PASS_ACTION, 24)
    self.assertEqual(euchre.CLUBS_TRUMP_ACTION, 25)
    self.assertEqual(euchre.DIAMONDS_TRUMP_ACTION, 26)
    self.assertEqual(euchre.HEARTS_TRUMP_ACTION, 27)
    self.assertEqual(euchre.SPADES_TRUMP_ACTION, 28)
    self.assertEqual(euchre.GO_ALONE_ACTION, 29)
    self.assertEqual(euchre.PLAY_WITH_PARTNER_ACTION, 30)
    self.assertEqual(euchre.MAX_BIDS, 8)
    self.assertEqual(euchre.NUM_TRICKS, 5)
    self.assertEqual(euchre.FULL_HAND_SIZE, 5)
    self.assertEqual(euchre.FULL_GAME_WINNING_SCORE, 10)
    self.assertEqual(euchre.MAX_FULL_GAME_HANDS, 19)
    game = pyspiel.load_game('euchre')
    state = game.new_initial_state()
    self.assertEqual(state.num_cards_dealt(), 0)
    self.assertEqual(state.num_cards_played(), 0)
    self.assertEqual(state.num_passes(), 0)
    self.assertEqual(state.upcard(), pyspiel.INVALID_ACTION)
    self.assertEqual(state.discard(), pyspiel.INVALID_ACTION)
    self.assertEqual(state.trump_suit(), pyspiel.INVALID_ACTION)
    self.assertEqual(state.left_bower(), pyspiel.INVALID_ACTION)
    self.assertEqual(state.right_bower(), pyspiel.INVALID_ACTION)
    self.assertEqual(state.declarer(), pyspiel.PlayerId.INVALID)
    self.assertEqual(state.declarer_partner(), pyspiel.PlayerId.INVALID)
    self.assertEqual(state.first_defender(), pyspiel.PlayerId.INVALID)
    self.assertEqual(state.second_defender(), pyspiel.PlayerId.INVALID)
    self.assertIsNone(state.declarer_go_alone())
    self.assertEqual(state.lone_defender(), pyspiel.PlayerId.INVALID)
    self.assertEqual(state.active_players(), [True, True, True, True])
    self.assertEqual(state.dealer(), pyspiel.INVALID_ACTION)
    self.assertEqual(state.current_phase(), euchre.Phase.DEALER_SELECTION)
    self.assertEqual(state.current_trick_index(), 0)
    self.assertEqual(state.card_holder(), [None] * 24)
    self.assertEqual(euchre.card_rank(8), euchre.JACK_RANK)
    self.assertEqual(euchre.card_rank(8, euchre.Suit.CLUBS), 100)
    self.assertEqual(euchre.card_suit(8), euchre.Suit.CLUBS)
    self.assertEqual(euchre.card_suit(8, euchre.Suit.SPADES),
                     euchre.Suit.SPADES)
    self.assertEqual(euchre.card_string(8), 'CJ')
    trick = state.tricks()[state.current_trick_index()]
    self.assertEqual(trick.winning_card(), pyspiel.INVALID_ACTION)
    self.assertEqual(trick.led_suit(), euchre.Suit.INVALID_SUIT)
    self.assertEqual(trick.trump_suit(), euchre.Suit.INVALID_SUIT)
    self.assertFalse(trick.trump_played())
    self.assertEqual(trick.leader(), pyspiel.PlayerId.INVALID)
    self.assertEqual(trick.winner(), pyspiel.PlayerId.INVALID)
    self.assertEqual(trick.cards(), [pyspiel.INVALID_ACTION])
    trick = state.current_trick()
    self.assertEqual(trick.led_suit(), euchre.Suit.INVALID_SUIT)
    self.assertEqual(trick.trump_suit(), euchre.Suit.INVALID_SUIT)
    self.assertFalse(trick.trump_played())
    self.assertEqual(trick.leader(), pyspiel.PlayerId.INVALID)
    self.assertEqual(trick.winner(), pyspiel.PlayerId.INVALID)
    self.assertEqual(trick.cards(), [pyspiel.INVALID_ACTION])

  def test_full_game_bindings(self):
    game = pyspiel.load_game('euchre_full')
    self.assertEqual(game.information_state_tensor_shape(), [
        euchre.FULL_GAME_INFORMATION_STATE_TENSOR_SIZE
    ])
    state = game.new_initial_state()
    self.assertEqual(state.team_scores(), [0, 0])
    self.assertEqual(state.current_hand_number(), 0)
    self.assertEqual(state.current_dealer(), pyspiel.PlayerId.INVALID)
    self.assertEqual(state.current_hand_state().current_phase(),
                     euchre.Phase.DEALER_SELECTION)
    tensor = state.information_state_tensor(0)
    self.assertLen(tensor, euchre.FULL_GAME_INFORMATION_STATE_TENSOR_SIZE)
    self.assertEqual(tensor[0], 1)    # N/S score is 0.
    self.assertEqual(tensor[14], 1)   # E/W score is 0.
    self.assertEqual(tensor[28], 1)   # First hand, zero-based.
    self.assertEqual(sum(tensor[:47]), 3)
    self.assertEqual(sum(tensor[47:]), 0)

  def test_information_state_bidding_and_go_alone_bits(self):
    state = self._state_at_bidding()
    state.apply_action(euchre.PASS_ACTION)
    tensor = state.information_state_tensor(state.current_player())
    self.assertEqual(tensor[28:68].index(1), 4)
    self.assertEqual(tensor[68:72], [0, 0, 0, 0])

    pending = self._state_at_go_alone()
    pending_tensor = pending.information_state_tensor(pending.current_player())
    self.assertEqual(pending_tensor[68:72], [0, 0, 0, 0])

    partner = self._state_at_go_alone()
    partner.apply_action(euchre.PLAY_WITH_PARTNER_ACTION)
    partner_tensor = partner.information_state_tensor(partner.current_player())
    self.assertEqual(partner_tensor[68:72], [0, 0, 0, 1])

    alone = self._state_at_go_alone()
    alone.apply_action(euchre.GO_ALONE_ACTION)
    alone_tensor = alone.information_state_tensor(alone.current_player())
    self.assertEqual(alone_tensor[68:72], [1, 0, 0, 1])


if __name__ == '__main__':
  absltest.main()
