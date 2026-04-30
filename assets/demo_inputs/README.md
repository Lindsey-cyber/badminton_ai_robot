# Demo 输入数据说明

## 推荐测试视频（公开可下载）

### 羽毛球比赛视频
1. **BWF 官方 YouTube 频道**
   - 搜索：`BWF badminton rally` 下载 720p 视频
   - 命令：`yt-dlp -f 22 "https://www.youtube.com/watch?v=..." -o badminton_sample.mp4`

2. **TrackNet 公开数据集**
   - 来源：https://hackmd.io/@TUIK/rJkRW54cU
   - 内容：26 个比赛视频，含轨迹标注

### 人体姿态测试
1. **COCO 验证集图像**（用于验证姿态模型精度）
2. 任意包含完整人体的视频（人体在画面 1/4 以上大小即可）

## 使用方法

```bash
# 用下载的视频测试姿态检测
python scripts/demo_pose_inference.py --input assets/demo_inputs/badminton_sample.mp4

# 测试球体检测
python scripts/demo_badminton_detection.py --input assets/demo_inputs/badminton_sample.mp4

# 批量推理
python scripts/video_batch_infer.py --input-dir assets/demo_inputs/ --mode both
```

## 无视频时

使用内置生成器：
```bash
# 生成完整 Demo 视频（无需外部模型和视频）
python scripts/generate_pose_demo_video.py --benchmark-chart

# 运行双目测距模拟
python scripts/stereo_distance_demo.py --mode simulate
```
