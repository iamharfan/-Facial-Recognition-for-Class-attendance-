# -Facial-Recognition-for-Class-attendance-
In the age of technological advancements, facial recognition has emerged as a powerful tool for enhancing security, convenience, and efficiency across various applications. One such application is in automating class attendance, where traditional manual methods are time-consuming and prone to human error. 

This project aims to develop an automated facial recognition system for class attendance using the OpenMV camera and Edge Impulse platform. By leveraging computer vision and machine learning, the system recognizes the faces of individuals, logs their attendance in real-time, and saves this data in a CSV file for future reference. 

The need for an efficient, accurate, and reliable attendance system is becoming increasingly important, especially in educational settings where managing large student populations can be a challenge. Traditional methods, such as roll call or manual sign-ins, can be inefficient, leading to errors, delays and tampering with sign-ins. This facial recognition-based solution offers an innovative alternative, providing an automated, accurate, and seamless way to track attendance while minimizing the need for human intervention. 

The project utilizes Edge Impulse – a platform specifically designed for building and deploying machine learning models- alongside OpenMV camera. A custom facial recognition model was trained on a diverse dataset to ensure robustness and accuracy under varying lighting and environmental conditions. 

Through this approach, we aimed to create a robust system that could be easily deployed and scaled in real-world educational environments, enhancing both the efficiency and reliability of attendance tracking. Our facial recognition attendance system has the potential to be extended to broader applications such as employee time tracking, secure access control, event management, and customer verification, thus offering efficient, contactless solutions for a variety of industrial applications. 

![image](https://github.com/user-attachments/assets/e10a80f6-c275-4f50-a8cc-d2e76ff4cf78)

The raw dataset for this project was composed initially of photographs captured using a mobile device which provided high resolution images, and the final model utilised the OpenMV camera for collection of the images. Each team member over the duration of this project contributed between 15-30 images resulting in a dataset diverse in ethnicity (European, African, Asian) and facial features. The images were captured under varying conditions such as: 

Facial angles – Frontal and side profiles to simulate real use cases 

Environmental variations- Various lighting conditions and backgrounds 

These were introduced to increase data heterogeneity and simulate natural uncontrolled environment and human behaviour. This diverse dataset was essential in providing generalization and support training. 

![image](https://github.com/user-attachments/assets/91bcf6da-41fe-49ef-a16c-3dd61582519a)

![image](https://github.com/user-attachments/assets/94e4fa0c-8178-4879-bf4c-df6b1e892693)

There were many challenges experienced when carrying out this project.  

Dataset Size- This was one of the biggest challenges as it was a process of trial and error to figure out the perfect dataset size, we started off with 30 images getting to nearly 600 images and it performed the best about real-life testing when deployed on the device. The results from edge impulse may not show this improvement but this may be due to the changing lighting conditions in real-time and this type of testing was the most important. 

Timestamps- The device struggled to read accurate time and some libraries would not import this may be due to the lack of Wi-Fi connection to the device but the solution for this problem was to set the time manually before starting application as this is a proof of concept this could be resolved if more time was allowed or finding alternative resolutions if the Wi-Fi connection would not resolve this. 

 

Overall, the project was carried out successfully and the objectives of this project were achieved. The model performed well once deployed sometimes making errors, but this could be resolved with higher resolution equipment. Another improvement that could have been made is that we could have introduced more augmentation techniques such as zooming.  
