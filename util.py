# import chess
import chess
import chess.pgn
import chess.svg
import numpy as np

import pred as my_pred
import data as my_data

### draw chess.svg + graph arrows
# chess.svg.board(boards[-20],
#     arrows=[chess.svg.Arrow(e[0], e[1], color='#ffffff80' if e[2].color else '#00000080') for e in edges],
#     size=500,
# )

### draw nx.DiGraph
# labels = nx.get_node_attributes(g, 'piece')
# nx.draw(g, {i: divmod(i, 8)[::-1] for i in range(64)}, labels=labels)

def win_prob_chart(g: chess.pgn.Game, engine='stockfish'):
    engine = my_pred.init_engine(engine)
    boards = my_data.get_boards(game, skip_first_n=0, skip_last_n=0)
    win_prob = np.array([my_pred.win_pred(b, engine=engine, pov='white', time=0.01) for b in boards])
    fig, ax = plt.subplots()
    ax.set_xlim(0.0, len(win_prob))
    ax.set_ylim(0.0, 1.0)
    # ax.plot(win_prob)
    ax.stackplot(range(len(win_prob)), win_prob, 1 - win_prob, colors=['w', 'grey'])
    ax.axhline(y=0.5, linestyle='--', color='k')
    ax.set_title('Predicted Win Probability')
    ax.set_xlabel('Move #')
    ax.set_ylabel('Win Probability for White')
    ax.text(1, 0.02, f'Actual Result: White {"Win" if row["Result"] == 1 else "Loss"}')
    plt.show()

def svg(b: chess.Board, edges=[], size=500):
    # color='#ffffff80' if b.turn else '#00000080'
    return chess.svg.board(b,
        arrows=[chess.svg.Arrow(e[0], e[1], color='#00000080') for e in edges],
        size=size,
    )
    