import os
import subprocess

files = os.listdir('videos')
for file in files:
    audio = file.split('.')[0]
    subprocess.run(['ffmpeg', '-i', f'videos/{file}',f'audios/{audio}.mp3'])