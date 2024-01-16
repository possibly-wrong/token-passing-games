import unrank
from dataclasses import dataclass, field
import math
from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve, gmres

@dataclass
class Dreidel:
    """Dreidel game probabilities and expected values."""
    num_coins: int
    is_ev: bool = True # True=E(# turns), False=P(player 0 wins)
    half_odd: int = 1 # 0=take smaller half, 1=take larger half
    min_coins: int = 0 # 0=out when can't pay, 1=out when zero coins
    values: dict = field(default_factory=dict)
    num_players: int = 1

    def solve(self, num_players):
        """Solve for values[(turn, state)] for (up to) number of players."""
        assert(self.num_coins >= (self.min_coins + 1) * num_players)
        while self.num_players + 1 < num_players:
            self.solve(self.num_players + 1)
        self.num_players = num_players
        data, rows, cols, b = [], [], [], [] # Solve sparse Ax=b
        for row, (state, turn) in enumerate(self.all_states()):
            data.append(1)
            rows.append(row)
            cols.append(row)
            value = 1 if self.is_ev else 0
            for next_state, next_turn in self.next_states(state, turn):
                n = len(next_state) - 1
                if n <= 1: # Game over
                    value += 0 if self.is_ev or n == 0 else 0.25
                elif n < num_players: # Some players lose
                    value += 0.25 * self.values[(next_state, next_turn)]
                else:
                    data.append(-0.25)
                    rows.append(row)
                    cols.append(unrank.rank_weak_composition(
                        [c - self.min_coins for c in next_state[:-1]] +
                        [next_state[-1] - 1]) * self.num_players + next_turn)
            b.append(value)
        x, info = gmres(csr_array((data, (rows, cols))), b, tol=1e-7)
        assert(info == 0)
        for (state, turn), v in zip(self.all_states(), x):
            self.values[(state, turn)] = v
        
    def all_states(self):
        """Generate all game states with current number of players."""
        for rank in range(math.comb(self.num_coins - 1 -
                                    self.num_players * (self.min_coins - 1),
                                    self.num_players)):
            state = unrank.unrank_weak_composition(rank, self.num_players + 1,
                self.num_coins - 1 - self.num_players * self.min_coins)
            state = tuple([c + self.min_coins for c in state[:-1]] +
                          [state[-1] + 1])
            for turn in range(self.num_players):
                yield state, turn

    def next_states(self, state, turn):
        """Generate distribution of successor states."""

        # Nothing (Nun)
        yield self.remove_losers(list(state), turn)

        # Everything (Gimel)
        next_state = list(state)
        next_state[turn] += next_state[-1]
        next_state = [c - 1 for c in next_state[:-1]] + [self.num_players]
        yield self.remove_losers(next_state, turn)

        # Half (Hei)
        next_state = list(state)
        win = (next_state[-1] + self.half_odd) // 2
        next_state[turn] += win
        next_state[-1] -= win
        if next_state[-1] == 0:
            next_state = [c - 1 for c in next_state[:-1]] + [self.num_players]
        yield self.remove_losers(next_state, turn)
        
        # Put (Shin)
        next_state = list(state)
        next_state[turn] -= 1
        next_state[-1] += 1
        yield self.remove_losers(next_state, turn)

    def remove_losers(self, state, turn):
        """Remove players without enough coins."""
        lose = self.min_coins - 1
        if state[0] == lose and not self.is_ev:
            return (0,), 0
        while lose in state:
            state[-1] += lose
            player = state.index(lose)
            del state[player]
            if player <= turn:
                turn -= 1
        return tuple(state), (turn + 1) % (len(state) - 1)

if __name__ == '__main__':
    for num_players in range(2, 8):
        print(f'{num_players} players:')
        for start_coins in range(1, 16):
            if sum(n * math.comb(start_coins * num_players - 1 + n, n) * (n + 3)
                   for n in range(2, num_players + 1)) > 1e8:
                continue
            print(f'    {start_coins} coins per player:')
            start = tuple([start_coins - 1] * num_players + [num_players])
            for is_ev in [True, False]:
                dreidel = Dreidel(start_coins * num_players, is_ev=is_ev)
                dreidel.solve(num_players)
                if is_ev:
                    ev = dreidel.values[(start, 0)]
                    print(f'        E(# turns) = {ev}')
                else:
                    for player in range(num_players):
                        p = dreidel.values[(start, (-player) % num_players)]
                        print(f'        P(player {player} wins) = {p}')
