import ast
import re
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
import os

# Fundus image resize
FUNDUS_TARGET_HEIGHT = 512
FUNDUS_TARGET_WIDTH = 768

# Dataframe column names
COL_IMAGE_PATH = "fundus_file_name"
COL_VF_PLOT = "vf_raw_value_plot"
COL_LATERALITY = "fundus_laterality"
COL_MD = "vf_md"
COL_DOB = "vf_dob"
COL_TEST_DATE = "vf_test_date"
COL_PID = "vf_id"


class vf_with_fundus_dataset(Dataset):

    def __init__(self, dataframe, transform=None, image_root=None):
        self.dataframe = dataframe
        self.transform = transform
        self.image_root = image_root

    def __len__(self):
        return len(self.dataframe)
    
    # ------------------------------------------------------   1. 有关眼底图像相关操作   ------------------------------------------------------------
    def get_fundus_image(self, data_row):
        '''
        从data_row中提取眼底图像
        '''
        image_path = data_row[COL_IMAGE_PATH]
        if self.image_root is not None:
            image_path = os.path.join(self.image_root, str(image_path))
        image = Image.open(image_path).convert("RGB")
        # if data_row[COL_LATERALITY] == "L":
        image = image.transpose(Image.FLIP_LEFT_RIGHT)  # 因为文件夹里面都是左眼，需要全部翻转为右眼才行
        image = image.resize((FUNDUS_TARGET_WIDTH, FUNDUS_TARGET_HEIGHT), Image.BILINEAR)
        if self.transform is not None:
            return self.transform(image)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        return image
    
    def normalize_fundus_image(self, image_tensor):
        '''
        正态归一化眼底图像（若不需要可删）
        '''
        mean = torch.mean(image_tensor)
        std = torch.std(image_tensor)
        image_tensor = (image_tensor - mean) / std
        return image_tensor
    
    # -------------------------------------------------------   2. 有关VF相关操作   -------------------------------------------------------------
    def clean_VF_in_table_to_list(self, data_row):
        '''
        从表格中安全提取出所有VF为列表
        '''
        # 1. 安全地将字符串转换为 Python 列表
        # ast.literal_eval 比 eval() 更安全，专门用于处理这种数据结构
        grid_VF = data_row[COL_VF_PLOT]
        rows = ast.literal_eval(grid_VF)
        extracted_values = []
        # 2. 遍历每一行字符串
        for row in rows:
            # 3. 使用正则表达式提取目标
            # (<0)  -> 匹配具体的字符串 "<0"s
            # |     -> 或者
            # (-?\d+) -> 匹配整数 (包含负号，虽然视野灵敏度通常为正，但为了通用性加上)
            matches = re.findall(r'(<0|-?\d+)', row)
            
            for val in matches:
                if val == '<0':
                    extracted_values.append(0)
                else:
                    extracted_values.append(int(val))
        return extracted_values
        
    def trun_VF_from_left_to_right(self, VF):
        '''
        将左眼 VF 翻转为右眼 VF
        '''
        segments = [4, 6, 8, 9, 9, 8, 6, 4]
        start_idx = 0
        new_VF = []

        for segment_length in segments:
            segment = VF[start_idx:start_idx + segment_length]
            new_VF.extend(segment[::-1])  # 行内左右翻转
            start_idx += segment_length

        # 右眼生理盲点（删除第26、第35个值）
        for idx in sorted([25, 34], reverse=True):
            if 0 <= idx < len(new_VF):
                del new_VF[idx]

        return np.array(new_VF)

    
    def get_VF_tensor(self, VF):
        '''
        获取VF的tensor表示
        '''
        VF_tensor = torch.tensor(VF, dtype=torch.float32)
        return VF_tensor

    def remove_right_blind_spots(self, VF):
        '''
        删除右眼生理盲点（第26、第35个值）
        '''
        for idx in sorted([25, 34], reverse=True):
            if 0 <= idx < len(VF):
                del VF[idx]
        return VF

    def get_age_from_dates(self, data_row):
        '''
        根据出生日期和检查日期计算年龄（年）
        '''
        dob = data_row.get(COL_DOB, None)
        test_date = data_row.get(COL_TEST_DATE, None)
        if dob is None or test_date is None:
            return None
        try:
            dob = np.datetime64(dob)
            test_date = np.datetime64(test_date)
            age_years = (test_date - dob).astype("timedelta64[D]") / np.timedelta64(365, "D")
            return float(age_years)
        except Exception:
            return None
    
    # -------------------------------------------------------   获取单条数据   -------------------------------------------------------------
    def __getitem__(self, idx):
        data_row = self.dataframe.iloc[idx]
        # 1. 获取并处理眼底图像
        fundus_img = self.get_fundus_image(data_row)
        if self.transform is None:
            fundus_img = self.normalize_fundus_image(fundus_img)
        # 2. 获取并处理VF数据
        VF = self.clean_VF_in_table_to_list(data_row)
        if data_row[COL_LATERALITY] == 'L':
            VF = self.trun_VF_from_left_to_right(VF)
        else:
            VF = self.remove_right_blind_spots(VF)
        VF_tensor = self.get_VF_tensor(VF)
        # 3. 获取MD值
        md_value = torch.tensor(float(data_row[COL_MD]), dtype=torch.float32)
        # 4. 获取眼睛部位和年龄
        laterality = data_row[COL_LATERALITY]
        age = self.get_age_from_dates(data_row)
        if age is None:
            age = -1.0
        pid = data_row[COL_PID]
        return {
            'fundus_img': fundus_img,
            'VF_tensor': VF_tensor,
            'md_tensor': md_value
        }   
