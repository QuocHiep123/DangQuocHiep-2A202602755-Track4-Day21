# Báo cáo Day 6: Đánh giá độ nhạy Calibration Drift LiDAR-Camera và Giám sát Chất lượng Chiếu điểm

- **Họ tên:** Đặng Quốc Hiệp
- **MSSV:** 2A202602755
- **Lớp:** VinUni AI20K - Track 4 (Computer Vision & Robotics)
- **Link repo:** https://github.com/QuocHiep123/DangQuocHiep-2A202602755-Track4-Day21
- **Topic:** A — LiDAR-camera projection QA
- **Dataset:** data/kitti_mini, data/nuscenes_mini_subset, data/synthetic
- **Các frame đã dùng:** 000011, 000000, scene-0103_010

## 1. Claim

Góc lệch extrinsic yaw của LiDAR vượt quá 1.0° (~17.5 mrad) khiến tỷ lệ điểm LiDAR rơi đúng vào bounding box 2D của các vật thể ở khoảng cách xa (≥30m) sụt giảm nghiêm trọng từ 100% xuống dưới 8%, trong khi sai lệch tịnh tiến ngang ty lên tới 15 cm vẫn giữ được trên 73% số điểm ở cùng cự ly; hiện tượng lệch góc nguy hiểm này có thể phát hiện tự động thông qua Point-Cluster Bbox IoU (ngưỡng báo động < 0.45) và Canny Edge Alignment Score.

## 2. Evidence

[ĐIỀN]

## 3. Failure case

[ĐIỀN]

## 4. Khuyến nghị nếu triển khai thật

[ĐIỀN]

## 5. Cách chạy lại

```bash
[ĐIỀN]
```

## 6. Khai báo sử dụng AI

| Công cụ | Dùng cho việc gì | Bạn đã kiểm chứng thế nào |
|---|---|---|
| [ĐIỀN] | | |
