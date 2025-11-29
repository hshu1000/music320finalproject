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

            # Instrument assignment: instrument personN <name>
            elif line.startswith('instrument '):
                parts = line.split()
                if len(parts) >= 3:
                    person_token = parts[1]  # expect 'person1', 'person2', ...
                    if not person_token.startswith('person'):
                        print('[terminal] Usage: instrument personN <instrument_name>')
                        continue
                    idx_str = person_token[len('person'):]
                    if not idx_str.isdigit():
                        print('[terminal] Person index must be an integer, e.g. person1, person2')
                        continue
                    person_index = int(idx_str)
                    instr_name = ' '.join(parts[2:])  # e.g. 'piano'

                    fp.set_person_instrument(person_index, instr_name)
                    # set_person_instrument itself prints success / error
                else:
                    print('[terminal] Usage: instrument personN <instrument_name>')

            # Exit terminal thread
            elif line in ('quit', 'exit'):
                print('[terminal] Stopping terminal loop (synth continues).')
                break

            elif line == 'help':
                print('Commands:')
                print('  mode hand / mode arm')
                print('  scale <name>')
                print('  instrument personN <instrument_name>')
                print(f'     available instruments: {", ".join(fp.list_instruments())}')
                print('  quit / exit')

        except Exception as e:
            print(f'[terminal] Error reading command: {e}')
            break


def main():
    # Start matplotlib figure
    init_plot()

    # Start audio callback stream
    fp.start_audio_thread()

    # Start terminal control loop in a background thread
    t = threading.Thread(target=terminal_command_loop, daemon=True)
    t.start()

    # Start pose detection
    start_pose_detection()


if __name__ == '__main__':
    main()
