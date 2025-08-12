
# ffprobe常用命令

- 获取视频的帧数：
```bash
ffprobe -loglevel warning -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 input.mp4
```

- 获取视频FPS：
```bash
ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate,avg_frame_rate -of default=noprint_wrappers=1 input.mp4
```

- 查看视频帧的时间戳：
```bash
ffprobe -show_frames -select_streams v input.mp4 | grep "pkt_pts_time"
```

- 查看音频帧的时间戳：
```bash
ffprobe -show_frames -select_streams a input.mp4 | grep "pkt_pts_time"
```


# ffmpeg常用命令

- 给视频添加音频（来自aac或者mp3等音频文件）：
```bash
ffmpeg -i input.mp4 -map 0:a -c:a copy output.aac
```

- 给视频添加音频（来自另外的视频文件）：
```bash
ffmpeg -i silent_video_path -i audio_video_path -c:v copy -map 0:v -map 1:a -shortest output_video_path
```

给视频图像上采样为原来的1.5倍
```bash
ffmpeg -i input.mp4 -vf "scale=iw*1.5:ih*1.5" output.mp4
```

- 给视频图像上采样为原来的1.5倍（自动取整为偶数）：
```bash
ffmpeg -i input.mp4 -vf "scale=ceil(iw*1.5/2)*2:ceil(ih*1.5/2)*2" output.mp4
```



