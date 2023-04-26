import chess
import chess.engine

STOCKFISH_BIN = '/home/asy51/repos/Stockfish-sf_15.1/src/stockfish'
LC0_BIN = '/home/asy51/repos/lc0/build/release/lc0'

def init_engine(engine_str):
    if engine_str == 'stockfish':
        return chess.engine.SimpleEngine.popen_uci(STOCKFISH_BIN)
    elif engine_str == 'lc0':
        return chess.engine.SimpleEngine.popen_uci(LC0_BIN)
    else:
        ValueError

def win_pred(b: chess.Board, engine, pov='turn', time=0.1):
    """ties split in half to each player"""
    w = engine.analyse(b, chess.engine.Limit(time=time), info=chess.engine.INFO_SCORE)['score'].wdl().white().expectation()
    if pov == 'turn':
        if b.turn is False: return 1. - w
        return w
    elif pov == 'white':
        return w
    elif pov == 'black':
        return 1. - w
    else:
        raise ValueError