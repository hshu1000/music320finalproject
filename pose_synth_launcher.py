import threading

import freq_processing as fp
from pose_detect import start_pose_detection
from plotter import init_plot


def terminal_command_loop():
    import sys
    real_stdin = sys.__stdin__

    print('\n=== Pose Synth Terminal Control ===')
    print('Commands:')
    print('  mode hand                 -> use hand landmarks')
    print('  mode arm                  -> use YOLO arm pose')
    print('  scale <name>              -> set musical scale (e.g. "c major", "eb minor")')
    print('  instrument personN <name> -> set instrument for that person index')
    print(f'     available instruments: {", ".join(fp.list_instruments())}')
    print('  reverb on/off             -> toggle reverb')   # <<< ADDED
    print('  help                      -> show this help')
    print('  quit / exit               -> stop this listener (synth keeps running)')
    print('====================================\n')

    while True:
        try:
            line = real_stdin.readline()
            if not line:
                continue
            line = line.strip().lower()

            # Mode change
            if line.startswith('mode '):
                mode = line.split()[1]
                fp.set_global_mode(mode)
                print(f'[terminal] Mode set to {fp.CURRENT_MODE}')

            # Scale change
            elif line.startswith('scale '):
                scale = line[len('scale '):]
                fp.set_global_scale(scale)
                print(f'[terminal] Scale set to {fp.CURRENT_SCALE}')

            # Instrument assignment
            elif line.startswith('instrument '):
                parts = line.split()
                if len(parts) >= 3:
                    person_token = parts[1]
                    if not person_token.startswith('person'):
                        print('[terminal] Usage: instrument personN <instrument_name>')
                        continue
                    idx_str = person_token[len('person'):]
                    if not idx_str.isdigit():
                        print('[terminal] Person index must be an integer')
                        continue
                    person_index = int(idx_str)
                    instr_name = ' '.join(parts[2:])
                    fp.set_person_instrument(person_index, instr_name)
                else:
                    print('[terminal] Usage: instrument personN <instrument_name>')

            # === REVERB TOGGLE ADDED ===
            elif line == 'reverb on':
                fp.REVERB_ON = True
                print('[terminal] Reverb ENABLED')

            elif line == 'reverb off':
                fp.REVERB_ON = False
                print('[terminal] Reverb DISABLED')
            # ===========================

            elif line in ('quit', 'exit'):
                print('[terminal] Stopping terminal loop (synth continues).')
                break

            elif line == 'help':
                print('Commands:')
                print('  mode hand / mode arm')
                print('  scale <name>')
                print('  instrument personN <instrument_name>')
                print('  reverb on/off')
                print(f'     available instruments: {", ".join(fp.list_instruments())}')
                print('  quit / exit')

        except Exception as e:
            print(f'[terminal] Error reading command: {e}')
            break


def main():
    init_plot()
    fp.start_audio_thread()

    t = threading.Thread(target=terminal_command_loop, daemon=True)
    t.start()

    start_pose_detection()


if __name__ == '__main__':
    main()
