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

#include <array>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace euchre {
namespace {

void BasicGameTests() {
  testing::LoadGameTest("euchre");
  testing::ChanceOutcomesTest(*LoadGame("euchre"));
  testing::RandomSimTest(*LoadGame("euchre"), 10, /*serialize=*/true,
                         /*verbose=*/false);

  auto observer = LoadGame("euchre")
                      ->MakeObserver(kInfoStateObsType,
                                     GameParametersFromString("single_tensor"));
  testing::RandomSimTestCustomObserver(*LoadGame("euchre"), observer);

  std::shared_ptr<const Game> full_game = LoadGame("euchre_full");
  testing::LoadGameTest("euchre_full");
  testing::ChanceOutcomesTest(*full_game);
  testing::RandomSimTest(*full_game, 2, /*serialize=*/true,
                         /*verbose=*/false);

  auto full_game_observer = full_game->MakeObserver(
      kInfoStateObsType, GameParametersFromString("single_tensor"));
  testing::RandomSimTestCustomObserver(*full_game, full_game_observer);
}

void FullGameTerminalScoreTest() {
  std::shared_ptr<const Game> game = LoadGame("euchre_full");
  std::mt19937 rng(12345);
  for (int sim = 0; sim < 10; ++sim) {
    std::unique_ptr<State> state = game->NewInitialState();
    int actions_taken = 0;
    while (!state->IsTerminal()) {
      std::vector<Action> actions = state->LegalActions();
      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
      state->ApplyAction(actions[dis(rng)]);
      ++actions_taken;
      SPIEL_CHECK_LE(actions_taken, game->MaxGameLength());
    }

    const EuchreFullState* full_state =
        dynamic_cast<const EuchreFullState*>(state.get());
    SPIEL_CHECK_TRUE(full_state != nullptr);
    std::vector<int> team_scores = full_state->TeamScores();
    SPIEL_CHECK_EQ(team_scores.size(), 2);
    SPIEL_CHECK_TRUE(
        (team_scores[0] >= kFullGameWinningScore &&
         team_scores[1] < kFullGameWinningScore) ||
        (team_scores[1] >= kFullGameWinningScore &&
         team_scores[0] < kFullGameWinningScore));
    SPIEL_CHECK_LT(full_state->CurrentHandNumber(), kMaxFullGameHands);

    std::vector<double> returns = state->Returns();
    double ns_return = team_scores[0] > team_scores[1] ? 1.0 : -1.0;
    SPIEL_CHECK_EQ(returns[kNorth], ns_return);
    SPIEL_CHECK_EQ(returns[kSouth], ns_return);
    SPIEL_CHECK_EQ(returns[kEast], -ns_return);
    SPIEL_CHECK_EQ(returns[kWest], -ns_return);
  }
}

// Returns the cards that were never dealt to a player and are not the upcard.
std::set<int> Kitty(const State& state) {
  std::set<int> kitty;
  for (int card = 0; card < kNumCards; ++card) kitty.insert(card);
  std::vector<Action> history = state.History();
  for (int i = 1; i <= kNumPlayers * kNumTricks + 1; ++i) {
    kitty.erase(history[i]);
  }
  return kitty;
}

// Returns void_in[player][suit], inferred from the trick history.
std::array<std::array<bool, kNumSuits>, kNumPlayers> ObservedVoids(
    const EuchreState& state) {
  std::array<std::array<bool, kNumSuits>, kNumPlayers> void_in{};
  if (state.TrumpSuit() < 0) return void_in;
  Suit trump = static_cast<Suit>(state.TrumpSuit());
  std::vector<bool> active = state.ActivePlayers();
  for (const Trick& trick : state.Tricks()) {
    std::vector<int> cards = trick.Cards();
    if (trick.Leader() == kInvalidPlayer || cards.empty() ||
        cards[0] == kInvalidAction) {
      continue;
    }
    Suit led_suit = CardSuit(cards[0], trump);
    Player player = trick.Leader();
    for (int j = 1; j < cards.size(); ++j) {
      player = (player + 1) % kNumPlayers;
      while (!active[player]) player = (player + 1) % kNumPlayers;
      if (CardSuit(cards[j], trump) != led_suit) {
        void_in[player][static_cast<int>(led_suit)] = true;
      }
    }
  }
  return void_in;
}

void ResampleFromInfostateTest() {
  std::shared_ptr<const Game> game = LoadGame("euchre");
  std::mt19937 rng(12345);
  UniformProbabilitySampler sampler;
  int num_sims = 100;
  int num_kitty_resamples = 0;
  int num_kitty_differs = 0;
  int num_discard_resamples = 0;
  int num_discard_differs = 0;
  for (int sim = 0; sim < num_sims; ++sim) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      if (!state->IsChanceNode()) {
        const auto* euchre_state = static_cast<const EuchreState*>(state.get());
        std::array<absl::optional<Player>, kNumCards> holder =
            euchre_state->CardHolder();
        std::set<int> kitty = Kitty(*state);
        auto void_in = ObservedVoids(*euchre_state);
        Suit trump = static_cast<Suit>(euchre_state->TrumpSuit());
        for (int p = 0; p < state->NumPlayers(); ++p) {
          std::unique_ptr<State> resampled =
              state->ResampleFromInfostate(p, sampler);
          SPIEL_CHECK_EQ(state->InformationStateTensor(p),
                         resampled->InformationStateTensor(p));
          SPIEL_CHECK_EQ(state->InformationStateString(p),
                         resampled->InformationStateString(p));
          SPIEL_CHECK_EQ(state->CurrentPlayer(), resampled->CurrentPlayer());

          const auto* resampled_state =
              static_cast<const EuchreState*>(resampled.get());
          std::array<absl::optional<Player>, kNumCards> resampled_holder =
              resampled_state->CardHolder();
          for (int card = 0; card < kNumCards; ++card) {
            // The hero's current hand is unchanged.
            SPIEL_CHECK_EQ(holder[card] == p, resampled_holder[card] == p);
            // Hidden players' cards respect the voids they have shown.
            if (resampled_holder[card].has_value() &&
                *resampled_holder[card] != p &&
                euchre_state->TrumpSuit() >= 0) {
              int suit = static_cast<int>(CardSuit(card, trump));
              SPIEL_CHECK_FALSE(void_in[*resampled_holder[card]][suit]);
            }
          }

          ++num_kitty_resamples;
          if (Kitty(*resampled) != kitty) ++num_kitty_differs;
          if (euchre_state->Discard() != kInvalidAction &&
              p != euchre_state->Dealer()) {
            ++num_discard_resamples;
            if (resampled_state->Discard() != euchre_state->Discard()) {
              ++num_discard_differs;
            }
          } else {
            SPIEL_CHECK_EQ(resampled_state->Discard(), euchre_state->Discard());
          }
        }
      }
      std::vector<Action> actions = state->LegalActions();
      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
      state->ApplyAction(actions[dis(rng)]);
    }
  }
  // The kitty and a non-dealer's view of the discard are hidden information,
  // so they must actually be resampled.
  SPIEL_CHECK_GT(num_kitty_resamples, 0);
  SPIEL_CHECK_GT(num_kitty_differs, 0);
  SPIEL_CHECK_GT(num_discard_resamples, 0);
  SPIEL_CHECK_GT(num_discard_differs, 0);
  std::cout << "ResampleFromInfostateTest: kitty differs in "
            << num_kitty_differs << "/" << num_kitty_resamples
            << " resamples, non-dealer discard differs in "
            << num_discard_differs << "/" << num_discard_resamples
            << " resamples." << std::endl;
}

void FullGameResampleFromInfostateTest() {
  std::shared_ptr<const Game> game = LoadGame("euchre_full");
  std::mt19937 rng(12345);
  UniformProbabilitySampler sampler;
  int num_sims = 2;
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

}  // namespace
}  // namespace euchre
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::euchre::BasicGameTests();
  open_spiel::euchre::FullGameTerminalScoreTest();
  open_spiel::euchre::ResampleFromInfostateTest();
  open_spiel::euchre::FullGameResampleFromInfostateTest();
}
