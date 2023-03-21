# STAPose3D
Abstract: General movement and pose assessment of infants is an effective method for the early detection of cerebral palsy (CP). Nevertheless, most human pose estimation methods, whether in 2D or 3D, focus on adults due to the lack of large datasets and pose annotations on infants. To solve these problems, we presented a model known as YOLO-infantPose, which has been fine-tuned, for infant pose estimation in 2D. Additionally, we proposed a self-supervised model called STAPose3D for 3D infant pose estimation in videos. STAPose3D combines temporal convolution, temporal attention, and graph attention to jointly learn spatio-temporal features. Our method can be summarized into two stages: first applying YOLO-infantPose on input videos, and second lifting these 2D poses along with respective confidences for every joint to 3D. The employment of the best-performing 2D detector in the first stage can significantly improve the precision of 3D pose estimation. The results reveal that fine-tuned YOLO-infantPose outperforms all other models both on our clinical dataset as well as the public dataset MINI-RGBD. The results from our infant movement video dataset demonstrate that STAPose3D can effectively comprehend the spatio-temporal features among different views and significantly improve the performance of 3D infant pose estimation in videos.
