"""
가상 데이터센터 환경 데이터 생성 파이프라인
"""
#실행: (Green-IDC-Optimizer) ~/Cap1/Green-IDC-Optimizer$ python3 -m data.data_pipeline

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.config.enums import CoolingMode
from core.config.constants import FREE_COOLING_THRESHOLD_C, HYBRID_THRESHOLD_C
from domain.thermodynamics.cooling_load import(
    calculate_cooling_load_from_it_power_kw,
    calculate_cooling_load_from_airflow_kw,
    calculate_m_air_for_servers,    
)

from domain.thermodynamics.chiller import (
    calculate_chiller_power_kw,
    calculate_cop,
    ChillerResult,
    CoolingMode,
)

from domain.thermodynamics.free_cooling import (
    calculate_free_cooling,
    calculate_free_cooling_efficiency,
    FreeCoolingResult,
)

from domain.thermodynamics.it_power import (
    calculate_total_it_power_kw,
    calculate_server_power_w,
    ServerSpec,
    ServerType,
    CPU_SERVER,
    GPU_SERVER,
)

from domain.thermodynamics.pue import calculate_pue, PUEResult, PUE_BENCHMARK

lunar_year = [31,29,31,30,31,30,31,31,30,31,30,31]
norm_year = [31,28,31,30,31,30,31,31,30,31,30,31]

class Cluster:
    def __init__ (self):
        self.dff = pd.DataFrame()

    def read_file(self):
        df = pd.read_parquet('data/raw/cluster_trace_5min.parquet')
        print('Sorting by time...')

        df_sorted = df.sort_values('timestamp', ascending=True)
        #df_sorted.to_csv('data/raw/borg_traces_data.csv', index=False)
        print('Done. Shape:', df_sorted.shape)
        print('First 3 timestamps:', df_sorted['timestamp'].head(20).tolist())
        print('Last 3 timestamps:', df_sorted['timestamp'].tail(21).tolist())
        self.dff = df

    def extend_to_year(self, target_days=366):
    #28일 데이터를 1년으로 확장 (반복 + 노이즈 + 계절 트렌드)
    
        chunks = []
        repeats = target_days // 28 + 1  # 약 13번 반복
    
        for i in range(repeats):
            chunk = self.dff.copy()
        
            # 1) 가우시안 노이즈 추가 (±5~10%)
            noise = np.random.normal(1.0, 0.05, len(chunk))
            chunk['avg_cpu'] = (chunk['avg_cpu'] * noise).clip(0, 1)
            chunk['avg_mem'] = (chunk['avg_mem'] * noise).clip(0, 1)
            
            # 2) 계절 트렌드 추가 (여름에 부하 약간 상승)
            month = (i % 12) + 1  # 1~12월 매핑
            seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * (month - 1) / 12)
            # 7월(month=7)에 +10%, 1월(month=1)에 -10%
            chunk['avg_cpu'] = (chunk['avg_cpu'] * seasonal_factor).clip(0, 1)
            
            chunks.append(chunk)
    
        print(len(chunks))
        result = pd.concat(chunks, ignore_index=True)[:target_days * 288]  # 5분 단위
        # 3) 타임스탬프 재생성
        result['timestamp'] = pd.date_range(
            start='2024-01-01', periods=len(result), freq='5min'
        )
        print(result.head(3))
        print(result.tail(3))
        self.dff = result
    
    def extract_file(self):
        df_final = self.dff.copy()
        df_final.to_parquet("data/processed/cluster_trace_5min.parquet", index=False)
        print("저장 완료! →  data/processed/cluster_trace_5min.parquet")
        df_final.to_csv("data/processed/cluster_trace_5min.csv", index=False)
        print("저장 완료! →  data/processed/cluster_trace_5min.csv")

        print(df_final.head())
        print(df_final.tail())
        print(df_final.dtypes)

        # 잘 저장됐는지 확인
        df_check = pd.read_parquet("data/processed/cluster_trace_5min.parquet")
        print(f"불러오기 확인: {df_check.shape}")
        return df_final
    
    def cl_generate_dataset(self):
        self.read_file()
        self.extend_to_year(366)
        df = self.extract_file()
        #print(df.head(5))
        #print(df.tail(5))
        return df


class Weather:
    def __init__(self, year=2024, month=12, station_id=101):
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

        df_final.to_parquet("data/processed/weather.parquet", index=False)
        print("저장 완료! →  data/processed/weather.parquet")
        df_final.to_csv("data/processed/weather.csv", index=False)
        print("저장 완료! →  data/processed/weather.csv")

        print(df_final.head())
        print(df_final.tail())
        print(df_final.dtypes)

        # 잘 저장됐는지 확인
        df_check = pd.read_parquet("data/processed/weather.parquet")
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

#############

class SyntheticIDCBuilder:
    def __init__(self, num_servers=500, days=366):
        self.num_servers = num_servers
        self.days = days
        self.time_steps = days * 24 * 12  # 5분 단위
        self.cluster = Cluster()
        self.weather = Weather()
        
    def load_workload_pattern(self):
        """Google Cluster Trace에서 워크로드 패턴 로드"""
        df = self.cluster.cl_generate_dataset()
        # 시간대별 평균 CPU 사용률 패턴 추출
        return df
    
    def load_weather_data(self, year, station_id):
        """기상청 API에서 기상 데이터 로드"""
        # API 호출 로직
        months = 0
        days = 0
        for i in range(12):
            if year % 4 == 0:
                days += lunar_year[months]
                months+=1
            else:
                days += norm_year[months]
                months+=1
        
        print("months", months)
        self.weather = Weather(year, months, station_id)
        return self.weather.generate_dataset()
    
    def load_spec_data(self):
        df = pd.read_parquet("data/raw/specpower.parquet")
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
        df.to_parquet("data/processed/spec.parquet", index=False)
        df.to_csv("data/processed/spec.csv", index=False)
        return df
    
    def calculate_it_power(self, P_idle, P_max, cpu_utilization):
        """SPECpower 공식 기반 IT 전력 계산"""
        return self.num_servers * (P_idle + (P_max - P_idle) * cpu_utilization)
    

    #(New) Calculate m_air from # of servers (domain/cooling_load.py)
    def calculate_m_air(self):
        return calculate_m_air_for_servers(self.num_servers)

    #(New) Calculate Cooling Load from IT Power (domain/cooling_load.py)
    def calculate_cooling_load_it(self, it_power):
        """IT 전력 -> 냉각 부하 계산"""
        return calculate_cooling_load_from_it_power_kw(it_power)
    
    #(New) Calculate Cooling Load from air flow (domain/cooling_load.py)
    def calculate_cooling_load_air(self, supply_temp, return_temp):
        """"공기유량 + 온도차이 -> 냉각 부하"""
        m_air = self.calculate_m_air()# kg/s (공기 유량)
        return calculate_cooling_load_from_airflow_kw(m_air, supply_temp, return_temp)
    
    #(New) Calculate Chiller Power info (domain/chiller.py)
    #cooling_load_from_it
    def calculate_chiller_power(self, cooling_load, outside_temp):
        """COP 기반 칠러 전력 계산"""
        return calculate_chiller_power_kw(cooling_load, outside_temp)
        #COP(float), Chiller Power kw(Float), Cooling Mode(Enum)
    
    #(New) Calculate Free Cooling info (domain/free_cooling.py)
    #cooling_load_from_air
    def calculate_fc(self, cooling_load, outside_temp, outside_humidity):
        return calculate_free_cooling(cooling_load, outside_temp, outside_humidity)
        #is_available(Bool), efficiency(Float), fan_power_kw(Float), effective_cooling_kw(Float)

    #(New) Calculate PUE info (domain/pue.py)
    #cooling_load_from_it
    def calculate_pue_for_data(self, it_power_kw, cooling_load):
        return calculate_pue(it_power_kw, cooling_load)
        #PUE(Float), total_power_kw(Float), efficiency_vs_benchmark(Float)

    def generate_dataset(self):
        """통합 데이터셋 생성"""
        timestamps = pd.date_range(
            start='2024-01-01', 
            periods=self.time_steps, 
            freq='5min'
        )

        spec_data = self.load_spec_data()
        P_idle = 200.0 #CPU Server Standard
        P_max = 500.0 #CPU Server Standard
        
        cpu_util = self.load_workload_pattern() #Google Cluster 시간대별 사용량
        weather_data = self.load_weather_data(2024, 101) #2024년 춘천 (5분 단위로 보간됨)
        weather_data = weather_data.set_index("timestamp").reindex(timestamps).interpolate(method="linear").reset_index()

        data = {
            'timestamp': timestamps,
            'cpu_utilization': cpu_util["avg_cpu"].values,
            'outside_temp_c': weather_data["outdoor_temp_c"].values,
            'outside_humidity_pct': weather_data["outdoor_humidity"].values,
        }
        
        df = pd.DataFrame(data)

        # 1. IT 전력 (kW)
        df['it_power_kw'] = df['cpu_utilization'].apply(
            lambda cpu: self.calculate_it_power(P_idle, P_max, cpu)
        ) / 1000

        # 2. 냉각 부하 — IT 전력 기반
        df['cooling_load_it_kw'] = df['it_power_kw'].apply(
            self.calculate_cooling_load_it
        )

        # 3. 냉각 부하 — 공기 유량 기반 (supply 18°C, return 27°C)
        df['cooling_load_air_kw'] = df['it_power_kw'].apply(
            lambda _: self.calculate_cooling_load_air(18.0, 27.0)
        )

        # 4. 칠러: chiller_power_kw + cooling_mode
        def calc_chiller(row):
            result = self.calculate_chiller_power(row['cooling_load_it_kw'], row['outside_temp_c'])
            return pd.Series({
                'chiller_power_kw': result.chiller_power_kw,
                'cooling_mode': result.cooling_mode,
            })

        df[['chiller_power_kw', 'cooling_mode']] = df.apply(calc_chiller, axis=1)

        # 5. 자연공조: free_cooling_available + free_cooling_efficiency + fan_power_kw + effective_cooling_kw
        def calc_fc(row):
            result = self.calculate_fc(
                row['cooling_load_air_kw'], row['outside_temp_c'], row['outside_humidity_pct']
            )
            return pd.Series({
                'free_cooling_available': result.is_available,
                'free_cooling_efficiency': result.efficiency,
                'fan_power_kw': result.fan_power_kw,
                'effective_cooling_kw': result.effective_cooling_kw,
            })

        df[['free_cooling_available', 'free_cooling_efficiency', 'fan_power_kw', 'effective_cooling_kw']] = df.apply(calc_fc, axis=1)

        # 6. PUE: pue + total_power_kw
        def calc_pue(row):
            result = self.calculate_pue_for_data(row['it_power_kw'], row['chiller_power_kw'])
            return pd.Series({
                'pue': result.pue,
                'total_power_kw': result.total_power_kw,
            })

        df[['pue', 'total_power_kw']] = df.apply(calc_pue, axis=1)

        return df

# 사용 예시
builder = SyntheticIDCBuilder(num_servers=500, days=366)
dataset = builder.generate_dataset()
print(dataset.head())
print(dataset.tail())
print(dataset.dtypes)
dataset.to_parquet('data/processed/synthetic_idc_1year.parquet')