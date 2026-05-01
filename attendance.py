import os
from datetime import datetime
import pandas as pd

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    folder = "attendance_logs"
    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, f"{today}.csv")

    # لو الملف مش موجود
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(file_path, index=False)

    # قراءة البيانات
    df = pd.read_csv(file_path)

    # ❌ منع التكرار في نفس اليوم
    if name in df["Name"].values:
        return "Already marked today ❌"

    # إضافة سجل جديد
    new_row = pd.DataFrame([[name, today, time_now]],
                           columns=["Name", "Date", "Time"])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(file_path, index=False)

    return "Marked successfully ✅"