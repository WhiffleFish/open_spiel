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
#include <iterator>
#include <map>
#include <random>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace euchre {
namespace {

const GameType kGameType{/*short_name=*/"euchre",
                         /*long_name=*/"Euchre",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kTerminal,
                         /*max_num_players=*/kNumPlayers,
                         /*min_num_players=*/kNumPlayers,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/false,
                         /*parameter_specification=*/
                         {
                             {"allow_lone_defender", GameParameter(false)},
                             {"stick_the_dealer", GameParameter(true)},
                         }};

const GameType kFullGameType{
    /*short_name=*/"euchre_full",
    /*long_name=*/"Euchre Full Game",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/false,
    /*parameter_specification=*/
    {
        {"allow_lone_defender", GameParameter(false)},
    }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new EuchreGame(params));
}

std::shared_ptr<const Game> FullGameFactory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new EuchreFullGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);
REGISTER_SPIEL_GAME(kFullGameType, FullGameFactory);

open_spiel::RegisterSingleTensorObserver single_tensor(kGameType.short_name);
open_spiel::RegisterSingleTensorObserver full_game_single_tensor(
    kFullGameType.short_name);

std::map<Suit, Suit> same_color_suit {
  {Suit::kClubs, Suit::kSpades}, {Suit::kSpades, Suit::kClubs},
  {Suit::kDiamonds, Suit::kHearts}, {Suit::kHearts, Suit::kDiamonds}};

}  // namespace

Suit CardSuit(int card, Suit trump_suit) {
  Suit suit = CardSuit(card);
  if (CardRank(card) == kJackRank && same_color_suit[suit] == trump_suit)
    suit = trump_suit;
  return suit;
}

// Highest rank belongs to right bower, then left bower, then usual ranking.
int CardRank(int card, Suit trump_suit) {
  int rank = CardRank(card);
  if (CardSuit(card) == trump_suit && rank == kJackRank) {
    rank = 100;  // Right bower (arbitrary value)
  } else if (CardSuit(card, trump_suit) == trump_suit && rank == kJackRank) {
    rank = 99;  // Left bower (arbitrary value)
  }
  return rank;
}

EuchreGame::EuchreGame(const GameParameters& params)
    : Game(kGameType, params),
      allow_lone_defender_(ParameterValue<bool>("allow_lone_defender")),
      stick_the_dealer_(ParameterValue<bool>("stick_the_dealer")) {}

EuchreState::EuchreState(std::shared_ptr<const Game> game,
                         bool allow_lone_defender, bool stick_the_dealer)
    : State(game),
      allow_lone_defender_(allow_lone_defender),
      stick_the_dealer_(stick_the_dealer) {}

EuchreFullState::EuchreFullState(std::shared_ptr<const Game> game,
                                 GameParameters hand_game_params)
    : State(game), hand_game_params_(std::move(hand_game_params)) {
  hand_game_ = LoadGame("euchre", hand_game_params_);
  StartHand(kInvalidPlayer);
}

EuchreFullState::EuchreFullState(const EuchreFullState& other)
    : State(other),
      hand_game_params_(other.hand_game_params_),
      hand_game_(other.hand_game_),
      hand_state_(std::unique_ptr<EuchreState>(
          dynamic_cast<EuchreState*>(other.hand_state_->Clone().release()))),
      team_scores_(other.team_scores_),
      returns_(other.returns_),
      hand_number_(other.hand_number_),
      current_dealer_(other.current_dealer_),
      is_terminal_(other.is_terminal_) {}

Player EuchreFullState::CurrentPlayer() const {
  if (is_terminal_) return kTerminalPlayerId;
  return hand_state_->CurrentPlayer();
}

std::string EuchreFullState::ActionToString(Player player,
                                            Action action) const {
  return hand_state_->ActionToString(player, action);
}

std::string EuchreFullState::ToString() const {
  std::string rv = absl::StrCat(
      "Full-game score: N/S ", team_scores_[0], " E/W ", team_scores_[1],
      "\nTarget: ", kFullGameWinningScore, "\nHand: ", hand_number_ + 1,
      "\n");
  if (is_terminal_) {
    absl::StrAppend(&rv, "Winner: ",
                    team_scores_[0] > team_scores_[1] ? "N/S" : "E/W",
                    "\n");
  }
  absl::StrAppend(&rv, hand_state_->ToString());
  return rv;
}

std::string EuchreFullState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return absl::StrCat("Full-game score: N/S ", team_scores_[0], " E/W ",
                      team_scores_[1], "\nHand: ", hand_number_ + 1, "\n",
                      hand_state_->InformationStateString(player));
}

std::string EuchreFullState::ObservationString(Player player) const {
  return InformationStateString(player);
}

void EuchreFullState::InformationStateTensor(Player player,
                                             absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(values.size(), kFullGameInformationStateTensorSize);
  std::fill(values.begin(), values.end(), 0.0);

  auto ptr = values.begin();
  ptr[std::min(team_scores_[0], kFullGameMaxScore)] = 1;
  ptr += kFullGameMaxScore + 1;
  ptr[std::min(team_scores_[1], kFullGameMaxScore)] = 1;
  ptr += kFullGameMaxScore + 1;
  ptr[std::min(hand_number_, kMaxFullGameHands - 1)] = 1;
  ptr += kMaxFullGameHands;

  std::vector<float> hand_values(kInformationStateTensorSize);
  hand_state_->InformationStateTensor(player, absl::MakeSpan(hand_values));
  std::copy(hand_values.begin(), hand_values.end(), ptr);
  ptr += kInformationStateTensorSize;
  SPIEL_CHECK_EQ(ptr, values.end());
}

std::unique_ptr<State> EuchreFullState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  std::unique_ptr<EuchreFullState> clone(new EuchreFullState(*this));
  clone->hand_state_ = std::unique_ptr<EuchreState>(dynamic_cast<EuchreState*>(
      hand_state_->ResampleFromInfostate(player_id, rng).release()));
  SPIEL_CHECK_TRUE(clone->hand_state_ != nullptr);
  return clone;
}

std::vector<Action> EuchreFullState::LegalActions() const {
  if (is_terminal_) return {};
  return hand_state_->LegalActions();
}

std::vector<std::pair<Action, double>> EuchreFullState::ChanceOutcomes() const {
  if (is_terminal_) return {};
  return hand_state_->ChanceOutcomes();
}

std::vector<int> EuchreFullState::TeamScores() const {
  return {team_scores_[0], team_scores_[1]};
}

void EuchreFullState::StartHand(Player dealer) {
  std::unique_ptr<State> state = hand_game_->NewInitialState();
  hand_state_ =
      std::unique_ptr<EuchreState>(dynamic_cast<EuchreState*>(state.release()));
  SPIEL_CHECK_TRUE(hand_state_ != nullptr);
  if (dealer != kInvalidPlayer) {
    current_dealer_ = dealer;
    hand_state_->ApplyAction(dealer);
  }
}

void EuchreFullState::StartNextHand() {
  SPIEL_CHECK_NE(current_dealer_, kInvalidPlayer);
  ++hand_number_;
  StartHand((current_dealer_ + 1) % kNumPlayers);
}

void EuchreFullState::ScoreCompletedHand() {
  std::vector<double> hand_returns = hand_state_->Returns();
  SPIEL_CHECK_EQ(hand_returns.size(), kNumPlayers);
  SPIEL_CHECK_EQ(hand_returns[kNorth], hand_returns[kSouth]);
  SPIEL_CHECK_EQ(hand_returns[kEast], hand_returns[kWest]);
  SPIEL_CHECK_EQ(hand_returns[kNorth], -hand_returns[kEast]);

  int ns_delta = static_cast<int>(hand_returns[kNorth]);
  int ew_delta = static_cast<int>(hand_returns[kEast]);
  if (ns_delta > 0) {
    team_scores_[0] += ns_delta;
  } else if (ew_delta > 0) {
    team_scores_[1] += ew_delta;
  }
}

void EuchreFullState::SetTerminalReturns() {
  double ns_return = team_scores_[0] > team_scores_[1] ? 1.0 : -1.0;
  returns_[kNorth] = ns_return;
  returns_[kSouth] = ns_return;
  returns_[kEast] = -ns_return;
  returns_[kWest] = -ns_return;
}

void EuchreFullState::DoApplyAction(Action action) {
  SPIEL_CHECK_FALSE(is_terminal_);
  bool selecting_initial_dealer =
      hand_state_->CurrentPhase() == Phase::kDealerSelection;

  hand_state_->ApplyAction(action);
  if (selecting_initial_dealer) {
    current_dealer_ = action;
    return;
  }
  if (!hand_state_->IsTerminal()) return;

  ScoreCompletedHand();
  if (team_scores_[0] >= kFullGameWinningScore ||
      team_scores_[1] >= kFullGameWinningScore) {
    is_terminal_ = true;
    SetTerminalReturns();
  } else {
    StartNextHand();
  }
}

EuchreFullGame::EuchreFullGame(const GameParameters& params)
    : Game(kFullGameType, params),
      allow_lone_defender_(ParameterValue<bool>("allow_lone_defender")) {
  hand_game_params_["allow_lone_defender"] =
      GameParameter(allow_lone_defender_);
  hand_game_params_["stick_the_dealer"] = GameParameter(true);
  hand_game_ = LoadGame("euchre", hand_game_params_);
}

std::string EuchreFullGame::ActionToString(Player player,
                                           Action action_id) const {
  return hand_game_->ActionToString(player, action_id);
}

int EuchreFullGame::MaxGameLength() const {
  return kMaxFullGameHands *
         (1 +                         // Dealer selection
          (kNumPlayers * kNumTricks) + // Deal hands
          1 +                          // Upcard
          (2 * kNumPlayers) +          // Max 2 rounds of bidding
          1 +                          // Dealer discard
          1 +                          // Declarer go alone?
          (2 * allow_lone_defender_) + // Defenders go alone? (optional)
          (kNumPlayers * kNumTricks)); // Play of hand
}

int EuchreFullGame::MaxChanceNodesInHistory() const {
  return 1 +  // Initial dealer selection
         kMaxFullGameHands *
             ((kNumPlayers * kNumTricks) +  // Deal hands
              1);                           // Upcard
}

std::string EuchreState::ActionToString(Player player, Action action) const {
  if (history_.empty()) return DirString(action);
  if (action == kPassAction) return "Pass";
  if (action == kClubsTrumpAction) return "Clubs";
  if (action == kDiamondsTrumpAction) return "Diamonds";
  if (action == kHeartsTrumpAction) return "Hearts";
  if (action == kSpadesTrumpAction) return "Spades";
  if (action == kGoAloneAction) return "Alone";
  if (action == kPlayWithPartnerAction) return "Partner";
  return CardString(action);
}

std::string EuchreState::ToString() const {
  std::string rv = "Dealer: ";
  absl::StrAppend(&rv, DirString(dealer_), "\n\n");
  absl::StrAppend(&rv, FormatDeal());
  if (upcard_ != kInvalidAction)
    absl::StrAppend(&rv, "\nUpcard: ", ActionToString(kInvalidPlayer, upcard_));
  if (history_.size() > kFirstBiddingActionInHistory)
    absl::StrAppend(&rv, FormatBidding());
  if (discard_ != kInvalidAction) {
    absl::StrAppend(&rv, "\nDealer discard: ",
                    ActionToString(kInvalidPlayer, discard_), "\n");
  }
  if (declarer_go_alone_.has_value()) {
    absl::StrAppend(&rv, "\nDeclarer go alone: ");
    if (declarer_go_alone_.value())
      absl::StrAppend(&rv, "true\n");
    else
      absl::StrAppend(&rv, "false\n");
    if (allow_lone_defender_) {
      absl::StrAppend(&rv, "\nDefender go alone: ");
      if (lone_defender_ != kInvalidPlayer)
        absl::StrAppend(&rv, "true\n");
      else
        absl::StrAppend(&rv, "false\n");
    }
  }
  if (num_cards_played_ > 0) absl::StrAppend(&rv, FormatPlay(), FormatPoints());
  return rv;
}

std::array<std::string, kNumSuits> EuchreState::FormatHand(
    int player, bool mark_voids) const {
  // Current hand, except in the terminal state when we use the original hand
  // to enable an easy review of the whole deal.
  auto deal = IsTerminal() ? initial_deal_ : holder_;
  std::array<std::string, kNumSuits> cards;
  for (int suit = 0; suit < kNumSuits; ++suit) {
    cards[suit].push_back(kSuitChar[suit]);
    cards[suit].push_back(' ');
    bool is_void = true;
    for (int rank = kNumCardsPerSuit - 1; rank >= 0; --rank) {
      if (player == deal[Card(Suit(suit), rank)]) {
        cards[suit].push_back(kRankChar[rank]);
        is_void = false;
      }
    }
    if (is_void && mark_voids) absl::StrAppend(&cards[suit], "none");
  }
  return cards;
}

std::string EuchreState::FormatDeal() const {
  std::string rv;
  std::array<std::array<std::string, kNumSuits>, kNumPlayers> cards;
  for (auto player : {kNorth, kEast, kSouth, kWest})
    cards[player] = FormatHand(player, /*mark_voids=*/false);
  constexpr int kColumnWidth = 8;
  std::string padding(kColumnWidth, ' ');
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, padding, cards[kNorth][suit], "\n");
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, absl::StrFormat("%-8s", cards[kWest][suit]), padding,
                    cards[kEast][suit], "\n");
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, padding, cards[kSouth][suit], "\n");
  return rv;
}

std::string EuchreState::FormatBidding() const {
  SPIEL_CHECK_GE(history_.size(), kFirstBiddingActionInHistory);
  std::string rv;
  absl::StrAppend(&rv, "\nBidding:");
  absl::StrAppend(&rv, "\nNorth    East     South    West\n");
  if (dealer_ == 0) absl::StrAppend(&rv, absl::StrFormat("%-9s", ""));
  if (dealer_ == 1) absl::StrAppend(&rv, absl::StrFormat("%-18s", ""));
  if (dealer_ == 2) absl::StrAppend(&rv, absl::StrFormat("%-27s", ""));

  for (int i = kFirstBiddingActionInHistory; i < history_.size(); ++i) {
    if (i < kFirstBiddingActionInHistory + kNumPlayers - 1) {
      // Players can pass or "order up" the upcard to the dealer.
      if (history_[i].action == kPassAction)
        absl::StrAppend(&rv, absl::StrFormat("%-9s", "Pass"));
      else
        absl::StrAppend(&rv, absl::StrFormat("%-9s", "Order up!"));
    } else if (i == kFirstBiddingActionInHistory + kNumPlayers) {
      // Dealer can pass or "pick up" the upcard.
      if (history_[i].action == kPassAction)
        absl::StrAppend(&rv, absl::StrFormat("%-9s", "Pass"));
      else
        absl::StrAppend(&rv, absl::StrFormat("%-9s", "Pick up!"));
    } else {
      absl::StrAppend(
          &rv, absl::StrFormat(
               "%-9s", ActionToString(kInvalidPlayer, history_[i].action)));
    }
    if (history_[i].player == kNumPlayers - 1) rv.push_back('\n');
    if (history_[i].action > kPassAction) break;
  }

  absl::StrAppend(&rv, "\n");
  return rv;
}

std::string EuchreState::FormatPlay() const {
  SPIEL_CHECK_GT(num_cards_played_, 0);
  std::string rv = "\nTricks:";
  absl::StrAppend(&rv, "\nN  E  S  W  N  E  S");
  for (int i = 0; i <= (num_cards_played_ - 1) / num_active_players_; ++i) {
    Player player_id = tricks_[i].Leader();
    absl::StrAppend(&rv, "\n", std::string(3 * player_id, ' '));
    for (auto card : tricks_[i].Cards()) {
      absl::StrAppend(&rv, CardString(card), " ");
      player_id = (player_id + 1) % kNumPlayers;
      while (!active_players_[player_id]) {
        absl::StrAppend(&rv, "   ");
        player_id = (player_id + 1) % kNumPlayers;
      }
    }
  }
  return rv;
}

std::string EuchreState::FormatPoints() const {
  std::string rv;
  absl::StrAppend(&rv, "\n\nPoints:");
  for (int i = 0; i < kNumPlayers; ++i)
    absl::StrAppend(&rv, "\n", DirString(i), ": ", points_[i]);
  return rv;
}

std::string EuchreState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  if (IsTerminal()) return ToString();
  std::string rv = "Dealer: ";
  absl::StrAppend(&rv, DirString(dealer_), "\n");
  if (upcard_ != kInvalidAction)
    absl::StrAppend(&rv, "Upcard: ", CardString(upcard_), "\n");
  absl::StrAppend(&rv, "Hand:\n");
  auto cards = FormatHand(player, /*mark_voids=*/true);
  for (int suit = kNumSuits - 1; suit >= 0; --suit)
    absl::StrAppend(&rv, cards[suit], "\n");
  if (history_.size() > kFirstBiddingActionInHistory)
    absl::StrAppend(&rv, FormatBidding());
  if (declarer_go_alone_.has_value()) {
    absl::StrAppend(&rv, "Declarer go alone: ",
                    declarer_go_alone_.value() ? "true" : "false", "\n");
    if (allow_lone_defender_) {
      absl::StrAppend(&rv, "Defender go alone: ",
                      lone_defender_ != kInvalidPlayer ? "true" : "false",
                      "\n");
    }
  }
  if (num_cards_played_ > 0) absl::StrAppend(&rv, FormatPlay(), FormatPoints());
  return rv;
}

std::string EuchreState::ObservationString(Player player) const {
  return InformationStateString(player);
}

void EuchreState::InformationStateTensor(Player player,
                                         absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  std::fill(values.begin(), values.end(), 0.0);
  SPIEL_CHECK_EQ(values.size(), kInformationStateTensorSize);
  if (upcard_ == kInvalidAction) return;
  auto ptr = values.begin();
  // Dealer position
  ptr[static_cast<int>(dealer_)] = 1;
  ptr += kNumPlayers;
  // Upcard
  ptr[upcard_] = 1;
  ptr += kNumCards;
  // Bidding [Clubs, Diamonds, Hearts, Spades, Pass]
  for (int i = 0; i < num_passes_; ++i) {
    ptr[kNumSuits] = 1;
    ptr += (kNumSuits + 1);
  }
  if (num_passes_ == 2 * kNumPlayers) return;
  if (trump_suit_ != Suit::kInvalidSuit) {
    ptr[static_cast<int>(trump_suit_)] = 1;
  }
  ptr += (kNumSuits + 1);
  for (int i = 0; i < 2 * kNumPlayers - num_passes_ - 1; ++i)
    ptr += (kNumSuits + 1);
  // Go alone
  if (declarer_go_alone_.value_or(false)) ptr[0] = 1;
  if (lone_defender_ != kInvalidPlayer) {
    if (lone_defender_ == first_defender_) ptr[1] = 1;
    if (lone_defender_ == second_defender_) ptr[2] = 1;
  }
  if (declarer_go_alone_.has_value()) ptr[3] = 1;
  ptr += 4;
  // Current hand
  for (int i = 0; i < kNumCards; ++i)
    if (holder_[i] == player) ptr[i] = 1;
  ptr += kNumCards;
  // History of tricks, presented in the format: N E S W N E S
  int current_trick = std::min(num_cards_played_ / num_active_players_,
                               static_cast<int>(tricks_.size() - 1));
  for (int i = 0; i < current_trick; ++i) {
    Player leader = tricks_[i].Leader();
    ptr += leader * kNumCards;
    int offset = 0;
    for (auto card : tricks_[i].Cards()) {
      ptr[card] = 1;
      ptr += kNumCards;
      ++offset;
      while (!active_players_[(leader + offset) % kNumPlayers]) {
        ptr += kNumCards;
        ++offset;
      }
    }
    SPIEL_CHECK_EQ(offset, kNumPlayers);
    ptr += (kNumPlayers - leader - 1) * kNumCards;
  }
  Player leader = tricks_[current_trick].Leader();
  int offset = 0;
  if (leader != kInvalidPlayer) {
    auto cards = tricks_[current_trick].Cards();
    ptr += leader * kNumCards;
    for (auto card : cards) {
      ptr[card] = 1;
      ptr += kNumCards;
      ++offset;
      while (!active_players_[(leader + offset) % kNumPlayers]) {
        ptr += kNumCards;
        ++offset;
      }
    }
  }
  // Current trick may contain less than four cards.
  if (offset < kNumPlayers) {
    ptr += (kNumPlayers - offset) * kNumCards;
  }
  // Move to the end of current trick.
  ptr += (kNumPlayers - std::max(leader, 0) - 1) * kNumCards;
  // Skip over unplayed tricks.
  ptr += (kNumTricks - current_trick - 1) * kTrickTensorSize;
  SPIEL_CHECK_EQ(ptr, values.end());
}

void EuchreState::HistoryTensor(Player player,
                                absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(values.size(), kHistoryTensorSize);
  InformationStateTensor(player, values.subspan(0, kInformationStateTensorSize));
  std::fill(values.begin() + kInformationStateTensorSize, values.end(), 0.0);
  auto ptr = values.begin() + kInformationStateTensorSize;
  // Like the information state's own hand, hands are only encoded once the
  // upcard is dealt.
  if (upcard_ != kInvalidAction) {
    for (int i = 0; i < kNumCards; ++i) {
      if (holder_[i].has_value()) ptr[*holder_[i] * kNumCards + i] = 1;
    }
  }
  ptr += kNumPlayers * kNumCards;
  ptr[player] = 1;
}

std::vector<Action> EuchreState::LegalActions() const {
  switch (phase_) {
    case Phase::kDealerSelection:
      return DealerSelectionLegalActions();
    case Phase::kDeal:
      return DealLegalActions();
    case Phase::kBidding:
      return BiddingLegalActions();
    case Phase::kDiscard:
      return DiscardLegalActions();
    case Phase::kGoAlone:
      return GoAloneLegalActions();
    case Phase::kPlay:
      return PlayLegalActions();
    default:
      return {};
  }
}

std::vector<Action> EuchreState::DealerSelectionLegalActions() const {
  SPIEL_CHECK_EQ(history_.size(), 0);
  std::vector<Action> legal_actions;
  legal_actions.reserve(kNumPlayers);
  for (int i = 0; i < kNumPlayers; ++i) legal_actions.push_back(i);
  return legal_actions;
}

std::vector<Action> EuchreState::DealLegalActions() const {
  std::vector<Action> legal_actions;
  legal_actions.reserve(kNumCards - num_cards_dealt_);
  for (int i = 0; i < kNumCards; ++i) {
    if (!holder_[i].has_value()) legal_actions.push_back(i);
  }
  SPIEL_CHECK_GT(legal_actions.size(), 0);
  return legal_actions;
}

std::vector<Action> EuchreState::BiddingLegalActions() const {
  std::vector<Action> legal_actions;
  legal_actions.push_back(kPassAction);
  if (stick_the_dealer_ && num_passes_ == 2 * kNumPlayers - 1)
    legal_actions.pop_back();
  Suit suit = CardSuit(upcard_);
  if (num_passes_ < kNumPlayers) {
    switch (suit) {
      case Suit::kClubs:
        legal_actions.push_back(kClubsTrumpAction);
        break;
      case Suit::kDiamonds:
        legal_actions.push_back(kDiamondsTrumpAction);
        break;
      case Suit::kHearts:
        legal_actions.push_back(kHeartsTrumpAction);
        break;
      case Suit::kSpades:
        legal_actions.push_back(kSpadesTrumpAction);
        break;
      case Suit::kInvalidSuit:
        SpielFatalError("Suit of upcard is invalid.");
    }
  } else {
    switch (suit) {
      case Suit::kClubs:
        legal_actions.push_back(kDiamondsTrumpAction);
        legal_actions.push_back(kHeartsTrumpAction);
        legal_actions.push_back(kSpadesTrumpAction);
        break;
      case Suit::kDiamonds:
        legal_actions.push_back(kClubsTrumpAction);
        legal_actions.push_back(kHeartsTrumpAction);
        legal_actions.push_back(kSpadesTrumpAction);
        break;
      case Suit::kHearts:
        legal_actions.push_back(kClubsTrumpAction);
        legal_actions.push_back(kDiamondsTrumpAction);
        legal_actions.push_back(kSpadesTrumpAction);
        break;
      case Suit::kSpades:
        legal_actions.push_back(kClubsTrumpAction);
        legal_actions.push_back(kDiamondsTrumpAction);
        legal_actions.push_back(kHeartsTrumpAction);
        break;
      case Suit::kInvalidSuit:
        SpielFatalError("Suit of upcard is invalid.");
    }
  }
  return legal_actions;
}

std::vector<Action> EuchreState::DiscardLegalActions() const {
  std::vector<Action> legal_actions;
  for (int card = 0; card < kNumCards; ++card) {
    if (holder_[card] == current_player_ && card != upcard_) {
      legal_actions.push_back(card);
    }
  }
  SPIEL_CHECK_EQ(legal_actions.size(), kNumTricks);
  return legal_actions;
}

std::vector<Action> EuchreState::GoAloneLegalActions() const {
  std::vector<Action> legal_actions;
  legal_actions.push_back(kGoAloneAction);
  legal_actions.push_back(kPlayWithPartnerAction);
  return legal_actions;
}

std::vector<Action> EuchreState::PlayLegalActions() const {
  std::vector<Action> legal_actions;
  // Check if we can follow suit.
  if (num_cards_played_ % num_active_players_ != 0) {
    Suit led_suit = CurrentTrick().LedSuit();
    if (led_suit == trump_suit_) {
      for (int rank = 0; rank < kNumCardsPerSuit; ++rank) {
        if (holder_[Card(led_suit, rank)] == current_player_) {
          legal_actions.push_back(Card(led_suit, rank));
        }
      }
      if (holder_[left_bower_] == current_player_) {
        // Left bower belongs to trump suit.
        legal_actions.push_back(left_bower_);
      }
    } else {
      for (int rank = 0; rank < kNumCardsPerSuit; ++rank) {
        if (holder_[Card(led_suit, rank)] == current_player_ &&
            Card(led_suit, rank) != left_bower_) {
          legal_actions.push_back(Card(led_suit, rank));
        }
      }
    }
  }
  if (!legal_actions.empty()) {
    absl::c_sort(legal_actions);  // Sort required because of left bower.
    return legal_actions;
  }
  // Can't follow suit, so we can play any of the cards in our hand.
  for (int card = 0; card < kNumCards; ++card) {
    if (holder_[card] == current_player_) legal_actions.push_back(card);
  }
  return legal_actions;
}

std::vector<std::pair<Action, double>> EuchreState::ChanceOutcomes() const {
  std::vector<std::pair<Action, double>> outcomes;
  if (history_.empty()) {
    outcomes.reserve(kNumPlayers);
    const double p = 1.0 / kNumPlayers;
    for (int dir = 0; dir < kNumPlayers; ++dir) {
      outcomes.emplace_back(dir, p);
    }
    return outcomes;
  }
  int num_cards_remaining = kNumCards - num_cards_dealt_;
  outcomes.reserve(num_cards_remaining);
  const double p = 1.0 / num_cards_remaining;
  for (int card = 0; card < kNumCards; ++card) {
    if (!holder_[card].has_value()) outcomes.emplace_back(card, p);
  }
  return outcomes;
}

void EuchreState::DoApplyAction(Action action) {
  switch (phase_) {
    case Phase::kDealerSelection:
      return ApplyDealerSelectionAction(action);
    case Phase::kDeal:
      return ApplyDealAction(action);
    case Phase::kBidding:
      return ApplyBiddingAction(action);
    case Phase::kDiscard:
      return ApplyDiscardAction(action);
    case Phase::kGoAlone:
      return ApplyGoAloneAction(action);
    case Phase::kPlay:
      return ApplyPlayAction(action);
    case Phase::kGameOver:
      SpielFatalError("Cannot act in terminal states");
  }
}

void EuchreState::ApplyDealerSelectionAction(int selected_dealer) {
  SPIEL_CHECK_EQ(history_.size(), 0);
  dealer_ = selected_dealer;
  phase_ = Phase::kDeal;
}

void EuchreState::ApplyDealAction(int card) {
  if (num_cards_dealt_ == kNumPlayers * kNumTricks) {
    initial_deal_ = holder_;  // Preserve the initial deal for easy retrieval.
    upcard_ = card;
    ++num_cards_dealt_;
    phase_ = Phase::kBidding;
    current_player_ = (dealer_ + 1) % kNumPlayers;
  } else {
    holder_[card] = (dealer_ + num_cards_dealt_) % kNumPlayers;
    ++num_cards_dealt_;
  }
}

void EuchreState::ApplyBiddingAction(int action) {
  if (action == kPassAction) {
    ++num_passes_;
    if (num_passes_ == kNumPlayers * 2) {
      phase_ = Phase::kGameOver;
      current_player_ = kTerminalPlayerId;
    } else {
      current_player_ = (current_player_ + 1) % kNumPlayers;
    }
  } else {
    // Trump suit selected.
    declarer_ = current_player_;
    first_defender_ = (declarer_ + 1) % kNumPlayers;
    declarer_partner_ = (declarer_ + 2) % kNumPlayers;
    second_defender_ = (declarer_ + 3) % kNumPlayers;
    switch (action) {
      case kClubsTrumpAction:
        trump_suit_ = Suit::kClubs;
        break;
      case kDiamondsTrumpAction:
        trump_suit_ = Suit::kDiamonds;
        break;
      case kHeartsTrumpAction:
        trump_suit_ = Suit::kHearts;
        break;
      case kSpadesTrumpAction:
        trump_suit_ = Suit::kSpades;
        break;
      default:
        SpielFatalError("Invalid bidding action.");
    }
    right_bower_ = Card(trump_suit_, kJackRank);
    left_bower_ = Card(same_color_suit[trump_suit_], kJackRank);
    if (num_passes_ < kNumPlayers) {
      // Top card was ordered up to dealer in first round of bidding.
      holder_[upcard_] = dealer_;
      phase_ = Phase::kDiscard;
      current_player_ = dealer_;
    } else {
      // Trump suit selected in second round of bidding.
      phase_ = Phase::kGoAlone;
    }
  }
}

void EuchreState::ApplyDiscardAction(int card) {
  SPIEL_CHECK_TRUE(holder_[card] == current_player_);
  discard_ = card;
  holder_[card] = absl::nullopt;
  phase_ = Phase::kGoAlone;
  current_player_ = declarer_;
}

void EuchreState::ApplyGoAloneAction(int action) {
  if (declarer_go_alone_.has_value() && allow_lone_defender_) {
    if (action == kGoAloneAction) {
      lone_defender_ = current_player_;
      active_players_[(lone_defender_ + 2) % kNumPlayers] = false;
      --num_active_players_;
      phase_ = Phase::kPlay;
      current_player_ = (dealer_ + 1) % kNumPlayers;
      while (!active_players_[current_player_]) {
        current_player_ = (current_player_ + 1) % kNumPlayers;
      }
    } else if (action == kPlayWithPartnerAction) {
      if (current_player_ == (dealer_ + 1) % kNumPlayers ||
          current_player_ == (dealer_ + 2) % kNumPlayers) {
        current_player_ = (current_player_ + 2) % kNumPlayers;
      } else {
        phase_ = Phase::kPlay;
        current_player_ = (dealer_ + 1) % kNumPlayers;
        while (!active_players_[current_player_]) {
          current_player_ = (current_player_ + 1) % kNumPlayers;
        }
      }
    } else {
      SpielFatalError("Invalid GoAlone action.");
    }
  } else {
    if (action == kGoAloneAction) {
      declarer_go_alone_ = true;
      active_players_[declarer_partner_] = false;
      --num_active_players_;
    } else if (action == kPlayWithPartnerAction) {
      declarer_go_alone_ = false;
    } else {
      SpielFatalError("Invalid GoAlone action.");
    }
    if (allow_lone_defender_) {
      current_player_ = (dealer_ + 1) % kNumPlayers;
      if (current_player_ == declarer_ || current_player_ == declarer_partner_)
        current_player_ = (current_player_ + 1) % kNumPlayers;
    } else {
      phase_ = Phase::kPlay;
      current_player_ = (dealer_ + 1) % kNumPlayers;
      if (declarer_go_alone_.value() && current_player_ == declarer_partner_) {
        current_player_ = (current_player_ + 1) % kNumPlayers;
      }
    }
  }
}

void EuchreState::ApplyPlayAction(int card) {
  SPIEL_CHECK_TRUE(holder_[card] == current_player_);
  holder_[card] = absl::nullopt;
  if (num_cards_played_ % num_active_players_ == 0) {
    CurrentTrick() = Trick(current_player_, trump_suit_, card);
  } else {
    CurrentTrick().Play(current_player_, card);
  }
  // Update player and point totals.
  Trick current_trick = CurrentTrick();
  ++num_cards_played_;
  if (num_cards_played_ % num_active_players_ == 0) {
    current_player_ = current_trick.Winner();
  } else {
    current_player_ = (current_player_ + 1) % kNumPlayers;
    while (!active_players_[current_player_]) {
      current_player_ = (current_player_ + 1) % kNumPlayers;
    }
  }
  if (num_cards_played_ == num_active_players_ * kNumTricks) {
    phase_ = Phase::kGameOver;
    current_player_ = kTerminalPlayerId;
    ComputeScore();
  }
}

void EuchreState::ComputeScore() {
  SPIEL_CHECK_TRUE(IsTerminal());
  std::vector<int> tricks_won(kNumPlayers, 0);
  for (int i = 0; i < kNumTricks; ++i) {
    tricks_won[tricks_[i].Winner()] += 1;
  }
  int makers_tricks_won = tricks_won[declarer_] + tricks_won[declarer_partner_];
  int makers_score;
  if (makers_tricks_won >= 0 && makers_tricks_won <= 2) {
    if (lone_defender_ >= 0)
      makers_score = -4;
    else
      makers_score = -2;
  } else if (makers_tricks_won >= 3 && makers_tricks_won <= 4) {
    makers_score = 1;
  } else if (makers_tricks_won == 5) {
    if (declarer_go_alone_.value())
      makers_score = 4;
    else
      makers_score = 2;
  } else {
    SpielFatalError("Invalid number of tricks won by makers.");
  }
  for (Player i = 0; i < kNumPlayers; ++i) {
    if (i == declarer_ || i == declarer_partner_)
      points_[i] = makers_score;
    else
      points_[i] = -makers_score;
  }
}

std::vector<Trick> EuchreState::Tricks() const {
  return std::vector<Trick>(tricks_.begin(), tricks_.end());
}

Trick::Trick(Player leader, Suit trump_suit, int card)
    : winning_card_(card),
      led_suit_(CardSuit(card, trump_suit)),
      trump_suit_(trump_suit),
      trump_played_(trump_suit != Suit::kInvalidSuit &&
                    trump_suit == led_suit_),
      leader_(leader),
      winning_player_(leader),
      cards_{card} {}

// TODO(jhtschultz) Find a simpler way of computing this.
void Trick::Play(Player player, int card) {
  cards_.push_back(card);
  bool new_winner = false;
  if (winning_player_ == kInvalidPlayer) new_winner = true;
  if (CardSuit(card, trump_suit_) == trump_suit_) {
    trump_played_ = true;
    if (CardSuit(winning_card_, trump_suit_) == trump_suit_) {
      if (CardRank(card, trump_suit_) > CardRank(winning_card_, trump_suit_)) {
        new_winner = true;
      }
    } else {
      new_winner = true;
    }
  } else {
    if (CardSuit(winning_card_, trump_suit_) != trump_suit_ &&
        CardSuit(winning_card_, trump_suit_) == CardSuit(card, trump_suit_) &&
        CardRank(card, trump_suit_) > CardRank(winning_card_, trump_suit_)) {
      new_winner = true;
    }
  }
  if (new_winner) {
    winning_card_ = card;
    winning_player_ = player;
  }
}

// Samples a full state uniformly from the worlds consistent with player_id's
// information set (ignoring the other players' policies).
//
// From player_id's point of view, the following is public or private-known and
// is kept fixed: the dealer, player_id's own dealt cards (including its own
// discard if it is the dealer), the upcard, the bidding/go-alone actions and
// every card played so far (together with who played it). The unknown cards
// are the unplayed cards in the other players' hands, the three undealt kitty
// cards and, when player_id is not the dealer, the dealer's discard. These are
// pooled and redistributed among bins: one per hidden player (sized by how
// many unplayed cards that player holds, excluding a picked-up upcard, which is
// public), one for the dealer's discard and one for the kitty. A hidden player
// may not receive a card of a suit (using effective suits, so the left bower
// counts as trump) that they have shown to be void in by failing to follow
// suit. Sampling is done by rejection over uniform shuffles of the pool, with a
// randomized backtracking search as a fallback when the void constraints make
// rejection impractically slow.
std::unique_ptr<State> EuchreState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  SPIEL_CHECK_TRUE(phase_ == Phase::kBidding || phase_ == Phase::kDiscard ||
                   phase_ == Phase::kGoAlone || phase_ == Phase::kPlay ||
                   phase_ == Phase::kGameOver);
  SPIEL_CHECK_GE(player_id, 0);
  SPIEL_CHECK_LT(player_id, kNumPlayers);

  // Detect void suits: scan trick history to find players who didn't follow
  // the led suit. Uses CardSuit(card, trump_suit_) to correctly treat the
  // left bower as belonging to the trump suit.
  std::array<std::array<bool, kNumSuits>, kNumPlayers> void_in{};
  if (num_cards_played_ > 0 && trump_suit_ != Suit::kInvalidSuit) {
    int play_start = history_.size() - num_cards_played_;
    int cards_per_trick = num_active_players_;
    int num_complete_tricks = num_cards_played_ / cards_per_trick;
    int cards_in_partial = num_cards_played_ % cards_per_trick;
    int num_tricks_to_scan =
        num_complete_tricks + (cards_in_partial > 0 ? 1 : 0);
    for (int t = 0; t < num_tricks_to_scan; ++t) {
      int trick_size =
          (t < num_complete_tricks) ? cards_per_trick : cards_in_partial;
      Player leader = tricks_[t].Leader();
      int first_card = history_[play_start].action;
      Suit led_suit = CardSuit(first_card, trump_suit_);
      Player player = leader;
      for (int j = 1; j < trick_size; ++j) {
        player = (player + 1) % kNumPlayers;
        while (!active_players_[player]) {
          player = (player + 1) % kNumPlayers;
        }
        int card = history_[play_start + j].action;
        if (CardSuit(card, trump_suit_) != led_suit) {
          void_in[player][static_cast<int>(led_suit)] = true;
        }
      }
      play_start += trick_size;
    }
  }

  // The dealer's discard is private to the dealer.
  const bool hidden_discard =
      discard_ != kInvalidAction && player_id != dealer_;

  // Cards whose location is known to player_id: its own dealt cards, the
  // upcard, and cards played by other players (with the player who played
  // them). Everything else goes into the pool of unknown cards.
  std::vector<std::vector<int>> player_cards(kNumPlayers);
  std::array<bool, kNumCards> in_pool{};
  in_pool.fill(true);
  in_pool[upcard_] = false;
  for (int card = 0; card < kNumCards; ++card) {
    absl::optional<Player> p = initial_deal_[card];
    if (!p.has_value()) continue;  // Upcard or kitty.
    if (*p == player_id ||
        (!holder_[card].has_value() && card != discard_)) {
      player_cards[*p].push_back(card);
      in_pool[card] = false;
    }
  }
  std::vector<int> pool;
  for (int card = 0; card < kNumCards; ++card) {
    if (in_pool[card]) pool.push_back(card);
  }

  // Bins to fill from the pool, in order: hidden players' unplayed cards,
  // the dealer's discard (if hidden) and the kitty.
  struct Bin {
    Player owner;  // kInvalidPlayer for the discard and kitty bins.
    int capacity;
  };
  std::vector<Bin> bins;
  for (Player p = 0; p < kNumPlayers; ++p) {
    if (p == player_id) continue;
    int hidden_count = 0;
    for (int card = 0; card < kNumCards; ++card) {
      if (holder_[card] == p) ++hidden_count;
    }
    if (p == dealer_ && holder_[upcard_] == p) --hidden_count;
    if (hidden_count > 0) bins.push_back({p, hidden_count});
  }
  const int discard_bin = hidden_discard ? bins.size() : -1;
  if (hidden_discard) bins.push_back({kInvalidPlayer, 1});
  bins.push_back({kInvalidPlayer, kNumCards - kNumPlayers * kNumTricks - 1});
  const int num_bins = bins.size();
  const int pool_size = pool.size();

  int total_capacity = 0;
  for (const Bin& bin : bins) total_capacity += bin.capacity;
  SPIEL_CHECK_EQ(pool_size, total_capacity);

  auto allowed = [&](int card, int bin) {
    Player owner = bins[bin].owner;
    if (owner == kInvalidPlayer || trump_suit_ == Suit::kInvalidSuit) {
      return true;
    }
    return !void_in[owner][static_cast<int>(CardSuit(card, trump_suit_))];
  };

  std::mt19937 shuffle_rng(rng() * std::mt19937::max());

  // bin_of[i] is the bin assigned to pool[i].
  std::vector<int> bin_of;
  bin_of.reserve(pool_size);
  for (int b = 0; b < num_bins; ++b) {
    bin_of.insert(bin_of.end(), bins[b].capacity, b);
  }

  // Rejection sampling: uniform over all consistent assignments.
  constexpr int kMaxRejectionAttempts = 1000;
  bool found = false;
  for (int attempt = 0; attempt < kMaxRejectionAttempts && !found; ++attempt) {
    std::shuffle(pool.begin(), pool.end(), shuffle_rng);
    found = true;
    for (int i = 0; i < pool_size; ++i) {
      if (!allowed(pool[i], bin_of[i])) {
        found = false;
        break;
      }
    }
  }

  // Fallback: randomized backtracking search, most-constrained cards first.
  // A solution always exists since the true state is consistent.
  if (!found) {
    std::shuffle(pool.begin(), pool.end(), shuffle_rng);
    std::vector<int> num_allowed(kNumCards, 0);
    for (int card : pool) {
      for (int b = 0; b < num_bins; ++b) num_allowed[card] += allowed(card, b);
    }
    std::stable_sort(pool.begin(), pool.end(), [&](int a, int b) {
      return num_allowed[a] < num_allowed[b];
    });
    std::vector<int> remaining(num_bins);
    for (int b = 0; b < num_bins; ++b) remaining[b] = bins[b].capacity;
    std::function<bool(int)> assign = [&](int i) -> bool {
      if (i == pool_size) return true;
      std::vector<int> candidates;
      for (int b = 0; b < num_bins; ++b) {
        if (remaining[b] > 0 && allowed(pool[i], b)) candidates.push_back(b);
      }
      std::shuffle(candidates.begin(), candidates.end(), shuffle_rng);
      for (int b : candidates) {
        --remaining[b];
        bin_of[i] = b;
        if (assign(i + 1)) return true;
        ++remaining[b];
      }
      return false;
    };
    found = assign(0);
  }
  SPIEL_CHECK_TRUE(found);

  int sampled_discard = discard_;
  for (int i = 0; i < pool_size; ++i) {
    int b = bin_of[i];
    if (bins[b].owner != kInvalidPlayer) {
      player_cards[bins[b].owner].push_back(pool[i]);
    } else if (b == discard_bin) {
      sampled_discard = pool[i];
      player_cards[dealer_].push_back(pool[i]);
    }
    // Otherwise the card is in the kitty and is simply never dealt.
  }
  for (Player p = 0; p < kNumPlayers; ++p) {
    SPIEL_CHECK_EQ(player_cards[p].size(), kNumTricks);
  }

  std::unique_ptr<State> clone = game_->NewInitialState();
  const EuchreState* clone_state = static_cast<EuchreState*>(clone.get());

  // Dealer selection is public.
  clone->ApplyAction(dealer_);

  // Deal the sampled hands in the same round-robin order as the original.
  for (int i = 0; i < kNumPlayers * kNumTricks; ++i) {
    Player deal_to = (dealer_ + i) % kNumPlayers;
    clone->ApplyAction(player_cards[deal_to].back());
    player_cards[deal_to].pop_back();
  }

  // Deal the same upcard (public knowledge).
  clone->ApplyAction(upcard_);

  // Replay all post-deal actions (bidding, discard, go-alone, play) from the
  // original history, substituting the sampled discard if it is hidden.
  int post_deal_start = 1 + kNumPlayers * kNumTricks + 1;
  for (size_t i = post_deal_start; i < history_.size(); ++i) {
    Action action = history_[i].action;
    if (hidden_discard && clone_state->CurrentPhase() == Phase::kDiscard) {
      SPIEL_CHECK_EQ(action, discard_);
      action = sampled_discard;
    }
    clone->ApplyAction(action);
  }

  SPIEL_CHECK_EQ(History().size(), clone->History().size());
  SPIEL_CHECK_EQ(State::InformationStateTensor(player_id),
                 clone->InformationStateTensor(player_id));
  return clone;
}

}  // namespace euchre
}  // namespace open_spiel
