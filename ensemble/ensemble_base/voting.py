import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import MinMaxScaler


class Ensemble:
    """
    [description]
    앙상블을 진행하는 클래스입니다.

    [parameter]
    files: 앙상블을 진행할 모델의 이름을 리스트 형태로 입력합니다.
    file_path: 앙상블을 진행할 모델의 csv 파일이 저장된 경로를 입력합니다.
    """

    def __init__(self, files: list, file_path: str):
        output_path = [
            file_path + file_name + ".csv" for file_name in files
        ]  # 앙상블 할 파일 경로 리스트
        self.output_list = [
            pd.read_csv(path) for path in output_path
        ]  # 앙상블 할 파일 리스트
        self.output_frame = pd.read_csv(output_path[0]).iloc[
            :, 0
        ]  # 결과(submission) 파일의 프레임

    def hard_voting(self):
        """
        [hard voting]
        전체 결과에 대해서, 유저별로 가장 많이 나온 아이템 10개를 뽑습니다.

        """
        # 각 output(submission)별로, 유저별 추천된 아이템 리스트 계산
        for i, output in enumerate(self.output_list):
            self.output_list[i] = (
                output.groupby(["user"])["item"].apply(list).reset_index()
            )

        # output별로 존재하는 유저별 아이템 리스트를 하나의 데이터프레임으로 병합
        print("##### merging all data ... ")
        items_by_user = self.output_list[0]
        for i in tqdm(range(len(items_by_user))):
            for df in self.output_list[1:]:
                items_by_user["item"][i].extend(df["item"][i])

        # 각 유저별로 가장 많이 나온 아이템 10개 추출
        items_by_user["item"] = items_by_user["item"].apply(
            lambda x: [item for item, cnt in Counter(x).most_common(10)]
        )

        # 결과 저장
        result = items_by_user.explode(column=["item"])

        return result

    def soft_voting(self, w_list: list):
        """
        [description]
        모델별로 item_score에 가중치를 곱하여, 가중된 score를 통해 상위 10개 아이템을 뽑습니다.

        [parameter]
        w_list: 모델별 가중치 리스트

        """
        # item 데이터프레임과 item_score 데이터프레임을 각각 생성
        df_item = pd.DataFrame()
        df_item_score = pd.DataFrame()

        for file in self.output_list:
            df_item = pd.concat([df_item, file["item"]], axis=1)
            df_item_score = pd.concat([df_item_score, file["score"]], axis=1)

        # 각 모델별로 item score 계산 방법이 다르므로 정규화 진행
        # scaler = MinMaxScaler()
        # df_scaled = scaler.fit_transform(df_item_score["score"])

        # weighted item score 계산
        weighted_scores = np.array(df_item_score["score"]) * np.array(w_list)

        # weighted item score가 최대인 모델 인덱스를 구함
        max_indices = np.argmax(weighted_scores, axis=1)

        # weighted item score가 최대인 모델의 아이템을 뽑음
        print("##### selecting ... ")
        selected_items = []
        for i in tqdm(range(df_item.shape[0])):
            selected_items.append(df_item.iloc[i, max_indices[i]])

        # 결과 저장
        result = pd.concat(
            [
                self.output_frame,
                pd.Series(selected_items),
            ],
            axis=1,
            keys=["user", "item"],
        )

        return result
