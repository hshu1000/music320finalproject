from pose_detect import start_pose_detection
from freq_processing import start_audio_thread
from plotter import init_plot

def main():
    init_plot()  # must be launched first
    start_audio_thread() 
    start_pose_detection()

if __name__ == "__main__":
    main()
