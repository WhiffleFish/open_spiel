// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "open_spiel/games/euchre/euchre.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace euchre {
namespace {

void CheckBits(const std::vector<float>& tensor, int offset, int size,
               const std::vector<int>& expected_ones) {
  for (int i = 0; i < size; ++i) {
    const bool should_be_one =
        std::find(expected_ones.begin(), expected_ones.end(), i) !=
        expected_ones.end();
    SPIEL_CHECK_EQ(tensor[offset + i], should_be_one ? 1.0f : 0.0f);
  }
}

void DealFixedHand(State* state, int dealer) {
  state->ApplyAction(dealer);
  for (int card = 0; card < kNumPlayers * kNumTricks; ++card) {
    state->ApplyAction(card);
  }
  state->ApplyAction(kNumPlayers * kNumTricks);
}

void BasicGameTests() {
  testing::LoadGameTest("euchre");
  testing::ChanceOutcomesTest(*LoadGame("euchre"));
  testing::RandomSimTest(*LoadGame("euchre"), 10);

  auto observer = LoadGame("euchre")
                      ->MakeObserver(kInfoStateObsType,
                                     GameParametersFromString("single_tensor"));
  testing::RandomSimTestCustomObserver(*LoadGame("euchre"), observer);
}

void ResampleFromInfostateTest() {
  std::shared_ptr<const Game> game = LoadGame("euchre");
  std::mt19937 rng(12345);
  UniformProbabilitySampler sampler;
  int num_sims = 100;
  for (int sim = 0; sim < num_sims; ++sim) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      if (!state->IsChanceNode()) {
        for (int p = 0; p < state->NumPlayers(); ++p) {
          std::unique_ptr<State> resampled =
              state->ResampleFromInfostate(p, sampler);
          SPIEL_CHECK_EQ(state->InformationStateTensor(p),
                         resampled->InformationStateTensor(p));
          SPIEL_CHECK_EQ(state->InformationStateString(p),
                         resampled->InformationStateString(p));
          SPIEL_CHECK_EQ(state->CurrentPlayer(), resampled->CurrentPlayer());
        }
      }
      std::vector<Action> actions = state->LegalActions();
      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
      state->ApplyAction(actions[dis(rng)]);
    }
  }
}

void PublicFeatureTensorTest() {
  std::shared_ptr<const Game> game = LoadGame("euchre");
  std::unique_ptr<State> state = game->NewInitialState();

  std::vector<float> initial_tensor = state->InformationStateTensor(2);
  SPIEL_CHECK_EQ(initial_tensor.size(), kInformationStateTensorSize);
  int offset = kLegacyInformationStateTensorSize;
  CheckBits(initial_tensor, offset, kNumPhases,
            {static_cast<int>(Phase::kDealerSelection)});
  offset += kNumPhases;
  CheckBits(initial_tensor, offset, kNumPlayers, {2});
  offset += kNumPlayers;
  CheckBits(initial_tensor, offset, kNumPlayers, {});
  offset += kNumPlayers;
  CheckBits(initial_tensor, offset, kNumTrumpTensorValues, {kNumSuits});
  offset += kNumTrumpTensorValues;
  CheckBits(initial_tensor, offset, kNumPlayers, {});
  offset += kNumPlayers;
  CheckBits(initial_tensor, offset, kNumPlayers, {});
  offset += kNumPlayers;
  CheckBits(initial_tensor, offset, kNumRoleTensorValues, {5});
  offset += kNumRoleTensorValues;
  CheckBits(initial_tensor, offset, kNumGoAloneStatusTensorValues, {0});

  DealFixedHand(state.get(), /*dealer=*/0);
  std::vector<float> bidding_tensor = state->InformationStateTensor(1);
  offset = kLegacyInformationStateTensorSize;
  CheckBits(bidding_tensor, offset, kNumPhases,
            {static_cast<int>(Phase::kBidding)});
  offset += kNumPhases;
  CheckBits(bidding_tensor, offset, kNumPlayers, {1});
  offset += kNumPlayers;
  CheckBits(bidding_tensor, offset, kNumPlayers, {1});
  offset += kNumPlayers;
  CheckBits(bidding_tensor, offset, kNumTrumpTensorValues, {kNumSuits});
  offset += kNumTrumpTensorValues + kNumPlayers + kNumPlayers;
  CheckBits(bidding_tensor, offset, kNumRoleTensorValues, {5});
  offset += kNumRoleTensorValues;
  CheckBits(bidding_tensor, offset, kNumGoAloneStatusTensorValues, {0});

  state->ApplyAction(kClubsTrumpAction);
  std::vector<float> discard_tensor = state->InformationStateTensor(1);
  offset = kLegacyInformationStateTensorSize;
  CheckBits(discard_tensor, offset, kNumPhases,
            {static_cast<int>(Phase::kDiscard)});
  offset += kNumPhases + kNumPlayers + kNumPlayers;
  CheckBits(discard_tensor, offset, kNumTrumpTensorValues,
            {static_cast<int>(Suit::kClubs)});
  offset += kNumTrumpTensorValues;
  CheckBits(discard_tensor, offset, kNumPlayers, {1});
  offset += kNumPlayers;
  CheckBits(discard_tensor, offset, kNumPlayers, {3});
  offset += kNumPlayers;
  CheckBits(discard_tensor, offset, kNumRoleTensorValues, {0, 3});
  offset += kNumRoleTensorValues;
  CheckBits(discard_tensor, offset, kNumGoAloneStatusTensorValues, {1});

  state->ApplyAction(0);
  std::unique_ptr<State> alone_state = state->Clone();
  alone_state->ApplyAction(kGoAloneAction);
  std::vector<float> alone_tensor = alone_state->InformationStateTensor(1);
  offset = kLegacyInformationStateTensorSize + kNumPhases + kNumPlayers +
           kNumPlayers + kNumTrumpTensorValues + kNumPlayers + kNumPlayers +
           kNumRoleTensorValues;
  CheckBits(alone_tensor, offset, kNumGoAloneStatusTensorValues, {3});

  state->ApplyAction(kPlayWithPartnerAction);
  std::vector<float> play_tensor = state->InformationStateTensor(1);
  offset = kLegacyInformationStateTensorSize;
  CheckBits(play_tensor, offset, kNumPhases,
            {static_cast<int>(Phase::kPlay)});
  offset += kNumPhases + kNumPlayers + kNumPlayers + kNumTrumpTensorValues +
            kNumPlayers + kNumPlayers + kNumRoleTensorValues;
  CheckBits(play_tensor, offset, kNumGoAloneStatusTensorValues, {2});

  std::vector<float> defender_tensor = state->InformationStateTensor(0);
  offset = kLegacyInformationStateTensorSize + kNumPhases;
  CheckBits(defender_tensor, offset, kNumPlayers, {0});
  offset += kNumPlayers;
  CheckBits(defender_tensor, offset, kNumPlayers, {0});
  offset += kNumPlayers + kNumTrumpTensorValues + kNumPlayers + kNumPlayers;
  CheckBits(defender_tensor, offset, kNumRoleTensorValues, {2, 4});
}

}  // namespace
}  // namespace euchre
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::euchre::BasicGameTests();
  open_spiel::euchre::ResampleFromInfostateTest();
  open_spiel::euchre::PublicFeatureTensorTest();
}
