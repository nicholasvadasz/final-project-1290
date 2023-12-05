import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSCI1290 - Final Project")
    parser.add_argument('-vi', '-video_input', help="Input video file")
    parser.add_argument('-o', '-output', help="Output video file")
    parser.add_argument('-fi', '-folder_input', help="Input folder")
    args = parser.parse_args()