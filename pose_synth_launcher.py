import threading

import freq_processing as fp
from pose_detect import start_pose_detection
from plotter import init_plot


def terminal_command_loop():
    import sys
    real_stdin = sys.__stdin__ 

    print('\n=== Pose Synth Terminal Control ===')
    print('Commands:')
    print('  mode hand           -> use hand landmarks')
    print('  mode arm            -> use YOLO arm pose')
    print('  scale <name>        -> set musical scale')
    print('  help                -> show this help')
    print('  quit / exit         -> stop this listener')
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

            # Exit terminal thread
            elif line in ('quit', 'exit'):
                print('[terminal] Stopping terminal loop (synth continues).')
                break

            elif line == 'help':
                print('Commands: mode hand/arm, scale <name>, quit')

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
