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
    print('  pedal on/off              -> sustain tail using echo/reverb')
    print('  pedal time <sec>          -> set sustain length (approx seconds)')
    print('  flanger on/off            -> toggle flanger effect')
    print('  flanger rate <Hz>         -> set flanger LFO rate')
    print('  flanger depth <ms>        -> set flanger modulation depth')
    print('  record start <name>       -> start WAV recording (recordings/)')
    print('  record stop               -> stop and save recording')
    print('  record status             -> show recording status')
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
                parts = line.split()
                if len(parts) >= 2:
                    mode = parts[1]
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

            # Pedal / sustain (also accept legacy "reverb" command)
            elif line in ('pedal on', 'reverb on'):
                fp.set_pedal_mode(True)

            elif line in ('pedal off', 'reverb off'):
                fp.set_pedal_mode(False)

            elif line.startswith('pedal time '):
                parts = line.split()
                if len(parts) >= 3:
                    fp.set_pedal_time(parts[2])
                elif len(parts) == 2:
                    fp.set_pedal_time(parts[1])
                else:
                    print('[terminal] Usage: pedal time <seconds>')

            # Flanger controls
            elif line == 'flanger on':
                fp.set_flanger_on(True)

            elif line == 'flanger off':
                fp.set_flanger_on(False)

            elif line.startswith('flanger rate '):
                parts = line.split()
                if len(parts) >= 3:
                    fp.set_flanger_params(rate=parts[2])
                elif len(parts) == 2:
                    fp.set_flanger_params(rate=parts[1])
                else:
                    print('[terminal] Usage: flanger rate <Hz>')

            elif line.startswith('flanger depth '):
                parts = line.split()
                if len(parts) >= 3:
                    fp.set_flanger_params(depth_ms=parts[2])
                elif len(parts) == 2:
                    fp.set_flanger_params(depth_ms=parts[1])
                else:
                    print('[terminal] Usage: flanger depth <ms>')

            # Recording controls
            elif line.startswith('record start'):
                parts = line.split(maxsplit=2)
                filename = parts[2] if len(parts) >= 3 else 'take.wav'
                fp.start_recording(filename)

            elif line == 'record stop':
                fp.stop_recording()

            elif line == 'record status':
                fp.recording_status()

            # Exit terminal thread
            elif line in ('quit', 'exit'):
                print('[terminal] Stopping terminal loop (synth continues).')
                break

            elif line == 'help':
                print('Commands:')
                print('  mode hand / mode arm')
                print('  scale <name>')
                print('  instrument personN <instrument_name>')
                print('  pedal on/off')
                print('  pedal time <sec>')
                print('  flanger on/off')
                print('  flanger rate <Hz>')
                print('  flanger depth <ms>')
                print('  record start <name>')
                print('  record stop')
                print('  record status')
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
