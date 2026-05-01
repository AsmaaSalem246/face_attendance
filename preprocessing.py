import cv2

def preprocess_image(img):
    # تقليل noise
    denoised = cv2.GaussianBlur(img, (5, 5), 0)

    # تحسين الإضاءة باستخدام LAB
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    enhanced_lab = cv2.merge((l, a, b))
    final_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return final_img