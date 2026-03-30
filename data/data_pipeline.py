"""
가상 데이터센터 환경 데이터 생성 파이프라인
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

lunar_year = [31,29,31,30,31,30,31,31,30,31,30,31]
norm_year = [31,28,31,30,31,30,31,31,30,31,30,31]

class Weather:
    def __init__(self, year=2019, month=3, station_id=101):
        self.year = year
        self.month = month
        self.station_id = station_id
        self.raw_list = []
        self.pre_list1 = []
        self.pre_list2 = []
        self.pre_list3 = []
        self.fin_list = []

    def make_response(self, numOfRows, startDt, endDt): #데이터 개수, 시작일자(YYYYMMDD), 끝일자
        SERVICE_KEY = "MFtQSWmriyJaWcgKOVr9Xuw2Gq%2BUhQa9dcGdEGDMK5pvvdptbr8Lc39CI5pW0xf4lgvwv7HzpzmyxDv4%2BG8KsA%3D%3D"

        url = (
            "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
            f"?serviceKey={SERVICE_KEY}" # encoding 인증 키
            f"&numOfRows={numOfRows}&pageNo=1" # 한 번에 받을 데이터 개수 (24시간, 24개), 페이지 번호
            "&dataType=JSON" # 응답 형식
            "&dataCd=ASOS&dateCd=HR" # 자료 코드(ASOS), 날짜 코드(HR)
            f"&startDt={startDt}&startHh=00" # 조회 시작일, 조회 시작 시각
            f"&endDt={endDt}&endHh=23" # 조회 종료일, 조회 종료 시각
            f"&stnIds={self.station_id}" # 관측 지점 번호
        )

        return requests.get(url)

    def make_df(self, resp):
        data = resp.json()

        # 응답 구조 확인
        items = data["response"]["body"]["items"]["item"]

        df = pd.DataFrame(items)

        # 프로젝트에서 쓸 컬럼만
        df = df[["tm", "ta", "hm", "ws"]].rename(columns={
            "tm": "timestamp",
            "ta": "outdoor_temp_c",        # 기온 (°C)
            "hm": "outdoor_humidity",    # 습도 (%)
            "ws": "outdoor_wind_speed",  # 풍속 (m/s)
        })

        df["outdoor_temp_c"] = pd.to_numeric(df["outdoor_temp_c"], errors="coerce")
        df["outdoor_humidity"] = pd.to_numeric(df["outdoor_humidity"], errors="coerce")
        df["outdoor_wind_speed"] = pd.to_numeric(df["outdoor_wind_speed"], errors="coerce")

        return df


    def preprocess_df(self, df):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df


    def process_null(self, df):
        print("=== 처리 전 결측치 ===")
        print(df.isnull().sum())

        # 기온: 선형 보간 (앞뒤 값으로 채우기, 최대 3시간 연속까지만)
        df["outdoor_temp_c"] = df["outdoor_temp_c"].interpolate(method="linear", limit=3)

        # 습도: 선형 보간
        df["outdoor_humidity"] = df["outdoor_humidity"].interpolate(method="linear", limit=3)

        # 풍속: 0으로 채우기 (바람 없는 걸로 간주)
        df["outdoor_wind_speed"] = df["outdoor_wind_speed"].fillna(0)

        # 그래도 남은 결측치 (3시간 넘게 연속 결측) → 앞/뒤 값으로 채우기
        df = df.ffill().bfill()

        print("\n=== 처리 후 결측치 ===")
        print(df.isnull().sum())

        return df

    def process_outlier(self, df):
        outlier_temp = df["outdoor_temp_c"][(df["outdoor_temp_c"] < -30) | (df["outdoor_temp_c"] > 50)]
        print(f"기온 이상치: {len(outlier_temp)}개")
        print(outlier_temp)

        df["outdoor_temp_c"] = df["outdoor_temp_c"].where(
            df["outdoor_temp_c"].between(-30, 50)  # 범위 밖은 NaN으로
        ).interpolate(method="linear")    # NaN은 보간으로 채우기

        # 습도: 0~100% 범위 강제
        df["outdoor_humidity"] = df["outdoor_humidity"].clip(0, 100)

        # 풍속: 음수 불가
        df["outdoor_wind_speed"] = df["outdoor_wind_speed"].clip(0, None)
        return df

    def resample_5min(self, df):
        """시간 단위 데이터를 5분 간격으로 선형 보간"""
        df = df.set_index("timestamp")
        df_5min = df.resample("5min").interpolate(method="linear")
        df_5min = df_5min.reset_index()
        return df_5min

    def out_file(self):
        df_final = pd.concat(self.fin_list, ignore_index=True)
        df_final = self.resample_5min(df_final)

        df_final.to_parquet("processed/weather.parquet", index=False)
        print("저장 완료! →  processed/weather.parquet")
        df_final.to_csv("processed/weather.csv", index=False)
        print("저장 완료! →  processed/weather.csv")

        print(df_final.head())
        print(df_final.tail())
        print(df_final.dtypes)

        # 잘 저장됐는지 확인
        df_check = pd.read_parquet("processed/weather.parquet")
        print(f"불러오기 확인: {df_check.shape}")
        return df_final

    def generate_dataset(self):
        for i in range(self.month): #0-Based month
            if self.year % 4 == 0:
                startDt = self.year*10000 + (i+1)*100 + 1
                endDt = startDt + lunar_year[i] - 1
                self.raw_list.append(self.make_response(lunar_year[i]*24, startDt, endDt))
            else:
                startDt = self.year*10000 + (i+1)*100 + 1
                endDt = startDt + norm_year[i] - 1
                self.raw_list.append(self.make_response(norm_year[i]*24, startDt, endDt))
        for i in range(self.month):
            self.pre_list1.append(self.make_df(self.raw_list[i]))
        for i in range(self.month):
            self.pre_list2.append(self.preprocess_df(self.pre_list1[i]))
        for i in range(self.month):
            self.pre_list3.append(self.process_null(self.pre_list2[i]))
        for i in range(self.month):
            self.fin_list.append(self.process_outlier(self.pre_list3[i]))
        return self.out_file()


class SyntheticIDCBuilder:
    def __init__(self, num_servers=500, days=90):
        self.num_servers = num_servers
        self.days = days
        self.time_steps = days * 24 * 12  # 5분 단위
        self.weather = Weather()
        
    def load_workload_pattern(self):
        """Google Cluster Trace에서 워크로드 패턴 로드"""
        df = pd.read_parquet("raw/cluster_trace_5min.parquet")
        # 시간대별 평균 CPU 사용률 패턴 추출
        hourly_pattern = df.groupby(df['timestamp'].dt.hour)['avg_cpu'].mean()
        return hourly_pattern
    
    def load_weather_data(self, year, station_id):
        """기상청 API에서 기상 데이터 로드"""
        # API 호출 로직
        months = 0
        days = 0
        for i in range(12):
            if year % 4 == 0:
                days += lunar_year[i]
            else:
                days += norm_year[i]
            if days > self.days:
                months = i+1
                break
        
        self.weather = Weather(year, months, station_id)
        return self.weather.generate_dataset()
    
    def load_spec_data(self):
        df = pd.read_parquet("raw/specpower.parquet")
        df.head()

        # 서버 카테고리 (mid만 남김)
        print(df["server_category"].value_counts())

        df = df[df["server_category"] == "mid"]

        #컬럼 확인 및 제거
        print(df.columns.to_list())

        df = df.drop(columns = ["vendor", "system", "server_category"])

        print(df)
        print(df.dtypes)
        print(df.count())

        # 내보내기
        df.to_parquet("processed/spec.parquet", index=False)
        df.to_csv("processed/spec.csv", index=False)
        return df
    
    def calculate_it_power(self, P_idle, P_max, cpu_utilization):
        """SPECpower 공식 기반 IT 전력 계산"""
        return self.num_servers * (P_idle + (P_max - P_idle) * cpu_utilization)
    
    def calculate_cooling_load(self, it_power, supply_temp, return_temp):
        """열역학 기반 냉각 부하 계산"""
        m_air = 50  # kg/s (공기 유량)
        c_p = 1.005  # kJ/kg·K (공기 비열)
        return m_air * c_p * (return_temp - supply_temp)
    
    def calculate_chiller_power(self, cooling_load, outside_temp):
        """COP 기반 칠러 전력 계산"""
        # COP = 6.0 - 0.1 * (outside_temp - 15)
        cop = max(2.0, 6.0 - 0.1 * (outside_temp - 15))
        return cooling_load / cop
    
    def generate_dataset(self):
        """통합 데이터셋 생성"""
        timestamps = pd.date_range(
            start='2024-01-01', 
            periods=self.time_steps, 
            freq='5min'
        )

        spec_data = self.load_spec_data()
        P_idle = spec_data["p_idle_w"].mean() # spec 데이터의 평균으로 계산
        P_max = spec_data["p_max_w"].mean() # spec 데이터의 평균으로 계산
        
        hourly_workload_pattern = self.load_workload_pattern() #Google Cluster 시간대별 사용량
        weather_data = self.load_weather_data(2024, 101) #2024년 춘천 (5분 단위로 보간됨)
        weather_data = weather_data.set_index("timestamp").reindex(timestamps).interpolate(method="linear").reset_index()

        data = {
            'timestamp': timestamps,
            'cpu_utilization': np.random.uniform(0.3, 0.8, self.time_steps),
            'outside_temp': weather_data["outdoor_temp_c"].values,
            'outside_humidity': weather_data["outdoor_humidity"].values,
        }
        
        df = pd.DataFrame(data)
        
        # 파생 변수 계산
        df['it_power_kw'] = df['cpu_utilization'].apply(lambda cpu: self.calculate_it_power(P_idle, P_max, cpu)) / 1000
        df['cooling_load_kw'] = df.apply(
            lambda x: self.calculate_cooling_load(x['it_power_kw'], 18, 27), axis=1
        )
        df['chiller_power_kw'] = df.apply(
            lambda x: self.calculate_chiller_power(x['cooling_load_kw'], x['outside_temp']), axis=1
        )
        df['pue'] = (df['it_power_kw'] + df['chiller_power_kw']) / df['it_power_kw']
        df['free_cooling_available'] = df['outside_temp'] < 15

        # 냉방도일 (Cooling Degree Days) — 기준온도 18°C
        BASE_TEMP = 18
        daily_avg_temp = df.groupby(df['timestamp'].dt.date)['outside_temp'].transform('mean')
        df['cooling_degree_days'] = (daily_avg_temp - BASE_TEMP).clip(lower=0)

        return df

# 사용 예시
builder = SyntheticIDCBuilder(num_servers=500, days=90)
dataset = builder.generate_dataset()
print(dataset.head())
print(dataset.tail())
print(dataset.dtypes)
dataset.to_parquet('processed/synthetic_idc_90days.parquet')