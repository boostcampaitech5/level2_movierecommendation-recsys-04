import os
import pandas as pd
import numpy as np
from collections import Counter

# from natsort import natsorted

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


class Ensemble:
    """
    [description]
    앙상블을 진행하는 클래스입니다.
    [parameter]
    files: 앙상블을 진행할 모델의 이름을 리스트 형태로 입력합니다.
    file_path: 앙상블을 진행할 모델의 csv 파일이 저장된 경로를 입력합니다.
    """

    def __init__(self, files: str, file_path: str):
        self.files = files
        self.output_list = []
        # self.item_list = []
        # self.item_score_list = []
        output_path = [
            file_path + file_name + "_sota.csv" for file_name in files
        ]
        self.output_frame = pd.read_csv(output_path[0]).iloc[:, 0]
        self.output_df = self.output_frame.copy()
        for path in output_path:
            # self.output_list.append(pd.read_csv(path)["item"].to_list())
            # self.item_list.append(pd.read_csv(path)["item"].to_list())
            # self.item_score_list.append(
            #     pd.read_csv(path)["item_score"].to_list()
            # )
            # self.output_df = pd.concat(
            #     [self.output_df, pd.read_csv(path).iloc[:, 1:]], axis=1
            # )
            self.output_list.append(pd.read_csv(path))
        # for file_name, output in zip(files, self.output_list):
        #     self.output_df[file_name] = output

    def hard_voting(self):
        """
        [hard voting] (다수결)
        전체 결과에 대해서, 유저별로 가장 많이 나온 아이템 10개를 뽑습니다.
        """
        # 각 output(submission)별로, 유저별 추천된 아이템 리스트 계산
        for i, output in enumerate(self.output_list):
            self.output_list[i] = (
                output.groupby(["user"])["item"].apply(list).reset_index()
            )
        # output별로 존재하는 유저별 아이템 리스트를 하나의 데이터프레임으로 병합
        print("##### merging all data")
        items_by_user = self.output_list[0]
        for i in tqdm(range(len(items_by_user))):
            for df in self.output_list[1:]:
                items_by_user["item"][i].extend(df["item"][i])
        # 각 유저별로 가장 많이 나온 아이템 10개 추출
        items_by_user["item"] = items_by_user["item"].apply(
            lambda x: [item for item, cnt in Counter(x).most_common(10)]
        )
        result = items_by_user.explode(column=["item"])
        return result

    def soft_voting(self):
        result = []
        # df = self.output_df
        # for file in file_list:
        #     f = pd.read_csv(file)
        #     df = pd.concat([df, f["item"], f["item_score"]], axis=1)
        # print(df)
        # print(df["item_score"])
        # # item_score => minmax 정규화
        # scaler = MinMaxScaler()
        # df_scaled = scaler.fit_transform(
        #     df["item_score"]
        # )  # df.iloc[:, 1:]  # user column 제외
        # df_scaled = pd.DataFrame(df_scaled)
        # print(df_scaled)
        # df_ws = pd.DataFrame()
        # for i in df_scaled:
        #     df_ws[i] = df_scaled.iloc[:, i] * w_list[i]
        # print(df_ws)
        return result
