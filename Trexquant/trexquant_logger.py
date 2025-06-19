import json
import csv
import time
from pathlib import Path

from notebooks.hangman_api_user import HangmanAPI

# --- HELPER FUNCTION ---
def format_game_summary(api):
    return {
        'game_id': api.game_id,
        'pattern_progression': api.pattern_progression,
        'guess_sequence': api.guessed_letters,
        'final_word': api.word,
        'win': api.won,
        'strategy_mode': api.agent.mode if hasattr(api, 'agent') else None,
        'guess_strategies': api.strategy_log if hasattr(api, 'strategy_log') else []
    }

# --- MAIN GAME LOOP ---
def run_trexquant_batch(agent_mode='Subpattern_Greedy',
                        num_games=100,
                        delay=3.1,
                        jsonl_file='trexquant_log.jsonl',
                        csv_file='trexquant_log.csv'):
    api = HangmanAPI(access_token="60f0e522873b5153cb4dbee0e497dd", timeout=2000)
    json_path = Path(jsonl_file)
    csv_path = Path(csv_file)

    with open(json_path, 'w') as jf, open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        csv_writer = csv.writer(cf)
        csv_writer.writerow(['Game ID', 'Win', 'Word', 'Guesses', 'Strategy Mode', 'Pattern Progression', 'Strategy Log'])

        for i in range(num_games):
            print(f"Playing game {i + 1}/{num_games}")
            game_data = {}
            try:
                api.start_game(practice=1, verbose=False)

                game_data = format_game_summary(api)
                jf.write(json.dumps(game_data) + '\n')
                jf.flush()

                csv_writer.writerow([
                    game_data['game_id'],
                    '✅' if game_data['win'] else '❌',
                    game_data['final_word'],
                    ', '.join(game_data['guess_sequence']),
                    game_data['strategy_mode'],
                    ' | '.join(game_data['pattern_progression']),
                    ' | '.join(game_data['guess_strategies'])
                ])
                cf.flush()
            except Exception as e:
                print(f"Error on game {i + 1}: {e}")
                game_data['error'] = str(e)
                jf.write(json.dumps(game_data) + '\n')
                jf.flush()
                csv_writer.writerow([
                    'ERROR', '❌', '', '', agent_mode, '', f'Exception: {e}'
                ])
                cf.flush()

            time.sleep(delay)

if __name__ == '__main__':
    run_trexquant_batch()
