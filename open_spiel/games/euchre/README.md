# Euchre

This directory contains two Euchre game registrations:

- `euchre`: one hand of Euchre.
- `euchre_full`: a full game of Euchre, played as repeated hands until one
  partnership reaches 10 points.

## Information State Tensor

### Single-hand `euchre`

The `euchre` information state tensor has shape `[935]`. Its layout is:

| Offset | Size | Field | Encoding |
| --- | ---: | --- | --- |
| `0` | 4 | Dealer | One-hot over `N, E, S, W`. |
| `4` | 24 | Upcard | One-hot over the 24-card Euchre deck. |
| `28` | 40 | Bidding | Eight slots of five bits each. Each slot is `[Clubs, Diamonds, Hearts, Spades, Pass]`. |
| `68` | 3 | Go-alone decisions | `[declarer_alone, first_defender_alone, second_defender_alone]`. Defender bits are only used when lone defenders are enabled. |
| `71` | 24 | Current hand | One-hot over the observer's current cards. |
| `95` | 840 | Trick history | Five trick records, each with seven 24-card lanes in `N E S W N E S` order. |

The current trick-history format uses seven lanes so a trick can be represented
from any leader while preserving table order. Unplayed or not-yet-public fields
remain zero. Before the upcard is known, the hand tensor after the dealer field
is all zeros.

### Full-game `euchre_full`

The `euchre_full` information state tensor has shape `[982]`. It is the
single-hand tensor with a 47-bit full-game prefix:

| Offset | Size | Field | Encoding |
| --- | ---: | --- | --- |
| `0` | 14 | N/S score | One-hot score bucket `0..13`. |
| `14` | 14 | E/W score | One-hot score bucket `0..13`. |
| `28` | 19 | Current hand number | One-hot zero-based hand index `0..18`. |
| `47` | 935 | Current hand state | The exact `euchre` information state tensor described above. |

The score buckets go to 13 because a partnership can start a final hand at
9 points and score up to 4 points. The hand-number field has 19 buckets because
with stick-the-dealer enabled, every hand awards at least one point to a
partnership; the longest possible game reaches 9-9 after 18 hands and ends on
hand 19.

`euchre_full` returns `+1` for each player on the winning partnership and `-1`
for each player on the losing partnership. The tensor contains the running
score because that score affects how close each partnership is to winning, even
though terminal utility is win/loss rather than score differential.
