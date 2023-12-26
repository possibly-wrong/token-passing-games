import unrank
from dataclasses import dataclass, field
import math
from itertools import product
import collections
from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve, gmres

@dataclass
class LeftCenterRight:
    """Left-Center-Right dice game probabilities and expected values."""
    num_players: int
    is_ev: bool = True # True=E(# turns), False=P(player 0 wins)
    max_rolls: int = 3
    values: dict = field(default_factory=dict)
    total_coins: int = 0

    def solve(self, total_coins):
        """Solve for values[(turn, state)] for (up to) total coins in play."""
        while self.total_coins + 1 < total_coins:
            self.solve(self.total_coins + 1)
        self.total_coins = total_coins
        data, rows, cols, b = [], [], [], [] # Solve sparse Ax=b
        for row, (state, turn) in enumerate(self.all_states(total_coins)):
            data.append(1)
            rows.append(row)
            cols.append(row)
            if state[turn] == total_coins:
                b.append(0 if self.is_ev or turn != 0 else 1)
                continue
            value = 1 if self.is_ev and state[turn] != 0 else 0
            counter = collections.Counter(self.next_states(state, turn))
            for (next_state, next_turn), count in counter.items():
                p = count / counter.total()
                next_total = sum(next_state)
                if next_state[next_turn] == next_total:
                    value += p if not self.is_ev and next_turn == 0 else 0
                elif next_total < total_coins:
                    value += p * self.values[(next_state, next_turn)]
                else:
                    data.append(-p)
                    rows.append(row)
                    cols.append(unrank.rank_weak_composition(next_state) *
                                self.num_players + next_turn)
            b.append(value)
        x, info = gmres(csr_array((data, (rows, cols))), b, tol=1e-7)
        assert(info == 0)
        for (state, turn), v in zip(self.all_states(total_coins), x):
            self.values[(state, turn)] = v
        
    def all_states(self, total_coins):
        """Generate all game states with given total coins in play."""
        for rank in range(math.comb(total_coins + self.num_players - 1,
                                    self.num_players - 1)):
            state = tuple(unrank.unrank_weak_composition(rank, self.num_players,
                                                         total_coins))
            for turn in range(self.num_players):
                yield state, turn

    def next_states(self, state, turn):
        """Generate distribution of successor states."""
        next_turn = (turn + 1) % self.num_players
        for roll in product([-1, self.num_players, 1, 0, 0, 0],
                            repeat=min(state[turn], self.max_rolls)):
            next_state = list(state)
            for die in roll:
                next_state[turn] -= 1
                if die < self.num_players:
                    next_state[(turn + die) % self.num_players] += 1
            yield tuple(next_state), next_turn

if __name__ == '__main__':
    num_coins = 3
    for num_players in range(2, 8):
        print(f'{num_players} players:')
        start = tuple([num_coins] * num_players)
        for is_ev in [True, False]:
            lcr = LeftCenterRight(num_players, is_ev=is_ev)
            lcr.solve(num_players * num_coins)
            if is_ev:
                ev = lcr.values[(start, 0)]
                print(f'    E(# turns) = {ev}')
            else:
                for player in range(num_players):
                    p = lcr.values[(start, (-player) % num_players)]
                    print(f'    P(player {player} wins) = {p}')
